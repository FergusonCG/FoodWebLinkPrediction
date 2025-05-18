import torch
import torch.nn as nn

class LinkPredictionModel(nn.Module):
    """
    Wrapper for link prediction models comprised of an encoder and decoder architecture.
    """
    def __init__(self, encoder, decoder):
        """
        Instantiate model.

        Parameters
        ----------
        encoder : torch.nn.Module
            A custom encoder architecture defined within one of encoders.py's classes. Can be `GCNEncoder`,
            `GATv2Encoder`, `GraphSAGEEncoder`, `TransformerEncoder`, or `SimpleEncoder`.
        decoder : torch.nn.Module
             A custom decoder architecture as a `LinkWeightDecoder` class from decoders.py.
        """
        super(LinkPredictionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index, edge_weight, decode_edges=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix.
        edge_index : torch.Tensor
            Graph connectivity (shape [2, num_edges]).
        edge_weight : torch.Tensor
            Edge weights corresponding to edge_index.
        decode_edges : torch.Tensor
            (Optional) Specific edges (shape [2, E_subset]) on which to run the decoder.
        
        Returns
        -------
        Union[torch.Tensor, Tuple(torch.Tensor, List[float])]
            If decode_edges is provided, returns a tuple (node_embeddings, pred_weights).
            Otherwise, returns node_embeddings as a torch.Tensor.
        """
        node_embeddings = self.encoder(x, edge_index, edge_weight)
        if decode_edges is not None:
            pred_weights = self.decoder(node_embeddings, decode_edges).squeeze(-1)
            return node_embeddings, pred_weights
        else:
            return node_embeddings
