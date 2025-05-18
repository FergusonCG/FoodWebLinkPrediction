from typing import Callable, List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data

from encoders import (
    GATv2Encoder,
    TransformerEncoder,
    GraphSAGEEncoder,
    GCNEncoder,
    SimpleEncoder
)
from decoders import LinkWeightDecoder
from models import LinkPredictionModel
from config import DEVICE


def train(
    model,
    food_web_graph,
    optimizer,
    loss_fn,
    train_idx=None,
    edge_dropout=0.0,
):
    """
    Single training step for the LinkPredictionModel.

    Parameters
    ----------
    model : models.LinkPredictionModel
        Instantiated LinkPredictionModel for which to train.
    food_web_graph : torch_geometric.data.Data
        PyG Data object containing the food web graph data to train the `model` on.
    optimizer : torch.optim.Optimizer
        Optimiser for the model, e.g. `torch.optim.Adam`.
    loss_fn : Callable
        Loss function, e.g. `torch.nn.L1Loss`).
    train_idx : torch.Tensor | None
        Indices of training edges within `food_web_graph` (default: None - trains on all edges).
    edge_dropout : float
        Probability for dropping each edge prior to message passing.

    Returns
    -------
    float
        MAE loss value for current mini‑batch.
    """
    model.train()
    optimizer.zero_grad()

    # Edge dropout
    edge_index = food_web_graph.edge_index
    edge_weights = food_web_graph.edge_weights
    if edge_dropout > 0.0:
        edge_index, mask = dropout_edge(
            edge_index,
            p=edge_dropout,
            force_undirected=True,
            training=True,
        )
        edge_weights = edge_weights[mask]
        if train_idx is not None:
            # Apply dropout, returns only non-zero edges that are not dropped
            train_idx = torch.nonzero(mask, as_tuple=False).view(-1)

    # Predict edge weights and calculate loss
    _emb, preds = model(food_web_graph.x, edge_index, edge_weights, decode_edges=edge_index)

    if train_idx is None or edge_dropout > 0.0:
        target = edge_weights  # already aligned with preds
    else:
        target = food_web_graph.edge_weights[train_idx]
        preds  = preds[train_idx]

    loss = loss_fn(preds, target)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
        model,
        food_web_graph,
        decode_edges,
        true_weights,
        loss_fn,
    ):
    """
    Evaluate the given `model` on a on a single food web Data object.

    Parameters
    ----------
    model : models.LinkPredictionModel
        Instantiated LinkPredictionModel for which to evaluate.
    food_web_graph : torch_geometric.data.Data
        PyG Data object containing the food web graph data to evaluate the `model` on.
    decode_edges : torch.Tensor
        Tensor containing target edges to predict in the val/test set.
    true_weights : torch.Tensor
        Ground‑truth weights for decode_edges.
    loss_fn
        Loss criterion for reporting, e.g. `torch.nn.L1Loss`.

    Returns
    -------
    mae : float
        Mean Absolute Error on this edge subset.
    preds : torch.Tensor
        Predicted weights (same shape as ``true_wts``).
    """
    model.eval()
    _embeddings, preds = model(food_web_graph.x, food_web_graph.edge_index, food_web_graph.edge_weights, decode_edges=decode_edges)
    mae = loss_fn(preds, true_weights).item()
    return mae, preds


def build_model(
        encoder_type,
        in_features,
        aggregator=None,
        heads=1,
        dropout=0.0,
        encoder_layer_sizes=1,
        decoder_layer_sizes=1,
        lr=0.01,
    ):
    """
    Create an instantiated LinkPredictionModel from the specified parameters.

    Parameters
    ----------
    encoder_type : str
        Type of encoder to use. Options are: "gatv2", "transformer", "graphsage", "gcn", or "simple".
    in_features : int
        Number of input features in raw food web data (the input dimension of the first encoder layer).
    aggregator : str | None
        Aggregator type for GraphSAGE encoder. Any aggregation of torch_geometric.nn.aggr can be used, 
        e.g., "mean", "max", or "lstm".
    heads : int
        Number of attention heads for GATv2 and Transformer encoders.
    dropout : float
        Dropout probability for GATv2 and Transformer encoders.
    encoder_layer_sizes : List[int] | int
        List of integers specifying the number hidden layers in the encoder, and how many neurons 
        to use in each layer.
    decoder_layer_sizes : List[int] | int
        List of integers specifying the number hidden layers in the decoder, and how many neurons 
        to use in each layer.
    lr : float
        Learning rate for the model's optimizer.
    """
    if isinstance(encoder_layer_sizes, int):
        encoder_layer_sizes = [encoder_layer_sizes]
    if isinstance(decoder_layer_sizes, int):
        decoder_layer_sizes = [decoder_layer_sizes]

    # Build encoder
    if encoder_type == "gatv2":
        encoder = GATv2Encoder(
            in_channels=in_features,
            layer_sizes=encoder_layer_sizes,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        final_dim = encoder_layer_sizes[-1] * heads
    elif encoder_type == "transformer":
        encoder = TransformerEncoder(
            in_channels=in_features,
            layer_sizes=encoder_layer_sizes,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
        )
        final_dim = encoder_layer_sizes[-1] * heads
    elif encoder_type == "graphsage":
        encoder = GraphSAGEEncoder(
            in_channels=in_features,
            layer_sizes=encoder_layer_sizes,
            aggr=aggregator or "mean",
        )
        final_dim = encoder_layer_sizes[-1]
    elif encoder_type == "gcn":
        encoder = GCNEncoder(
            in_channels=in_features,
            layer_sizes=encoder_layer_sizes,
        )
        final_dim = encoder_layer_sizes[-1]
    else: # If 'simple'
        encoder = SimpleEncoder()
        final_dim = in_features

    # Build decoder
    decoder = LinkWeightDecoder(final_dim, decoder_layer_sizes)

    # Create final model from encoder and decoder
    model = LinkPredictionModel(encoder, decoder).to(DEVICE)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.loss_fn = torch.nn.L1Loss()
    return model


def extract_layer_sizes(params, prefix, n_key):
    """
    Helper function to extract encoder and decoder layer architectures from HPO results.

    Parameters
    ----------
    params : Dict[str, int]
        Dictionary of model configuration parameters. Default in the form:
        {
            "n_encoder_layers": int,
            "encoder_layer_width_0": int,
            ...,
            "n_decoder_layers": int,
            "decoder_layer_width_0": int,
            ...,
            "lr": float,
            "encoder_type": str,
            "pca_components": int
        }
    prefix : str
        Prefix for the layer width keys in the params dictionary, e.g. "encoder_layer_width" 
        or "decoder_layer_width".
    n_key : str
        Key in the params dictionary that specifies the number of layers, e.g. "n_encoder_layers" 
        or "n_decoder_layers".
    
    Returns
    -------
    List[int]
        List of layer sizes for the encoder/decoder.
    """
    n = params[n_key]
    return [params[f"{prefix}_{i}"] for i in range(n)]


def model_from_params(params, in_features):
    """
    Instantiate a LinkPredictionModel from a configuration dictionary produced from HPO.

    Parameters
    ----------
    params : Dict
        Dictionary of model configuration parameters. Default in the form:
        {
            "n_encoder_layers": int,
            "encoder_layer_width_0": int,
            ...,
            "n_decoder_layers": int,
            "decoder_layer_width_0": int,
            ...,
            "lr": float,
            "encoder_type": str,
            "pca_components": int
        }
    in_features : int
        Number of input features in raw food web data (the input dimension of the first encoder layer).
    
    Returns
    -------
    models.LinkPredictionModel
        Instantiated LinkPredictionModel using the architecture defined in `params`.
    """
    encoder_type = params["encoder_type"]
    aggregator = params.get("graphsage_aggregator")
    heads = params.get("gatv2_heads", params.get("transformer_heads", 1))
    dropout = params.get("gatv2_dropout", params.get("transformer_dropout", 0.0))

    # Handle SimpleEncoder as special case as it does not actually perform embedding, hence more
    #   than a single layer is unecessary.
    if encoder_type == "simple":
        encoder_topology = [1]
        decoder_topology = extract_layer_sizes(params, "decoder_layer_width", "n_decoder_layers")
    else:
        encoder_topology = extract_layer_sizes(params, "encoder_layer_width", "n_encoder_layers")
        decoder_topology = extract_layer_sizes(params, "decoder_layer_width", "n_decoder_layers")

    return build_model(
        encoder_type=encoder_type,
        aggregator=aggregator,
        heads=heads,
        dropout=dropout,
        encoder_layer_sizes=encoder_topology,
        decoder_layer_sizes=decoder_topology,
        in_features=in_features,
        lr=params["lr"],
    )

# Metric definition
def ndcg_at_k(preds, true, k=10):
    """
    Calculate the normalised discounted cumulative gain (ndcg@k).
    
    Parameters
    ----------
    preds : torch.Tensor
        Predicted edge weights.
    true : torch.Tensor
        True edge weights.
    k : int
        Number of top predictions to consider for the NDCG calculation.

    Returns
    -------
    float
        Normalised discounted cumulative gain at `k`.
    """
    k = min(max(int(k), 1), preds.numel())
    _, idx_pred  = torch.topk(preds, k)
    _, idx_ideal = torch.topk(true,  k)

    gains_pred  = true[idx_pred]
    gains_ideal = true[idx_ideal]
    discounts   = torch.log2(torch.arange(k, device=true.device) + 2.0)

    dcg  = (gains_pred  / discounts).sum()
    idcg = (gains_ideal / discounts).sum()
    return (dcg / idcg).item() if idcg > 0 else 0.0


def r2_score(pred, true) -> float:
    """
    Calculate the R^2 score, measuring the proportion of variance explained by the model.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted edge weights.
    true : torch.Tensor
        True edge weights.

    Returns
    -------
    float
        R^2 score.
    """
    residual_sum_of_squares = torch.sum((pred - true) ** 2)
    total_sum_of_squares = torch.sum((true - true.mean()) ** 2) + 1e-8
    return 1.0 - (residual_sum_of_squares / total_sum_of_squares).item()


def mape(pred, true) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between predicted and true values.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted edge weights.
    true : torch.Tensor
        True edge weights.
    
    Returns
    -------
    float
        Mean Absolute Percentage Error (MAPE) as a percentage.
    """
    return (torch.mean(torch.abs((true - pred) / (true + 1e-8))) * 100).item()
