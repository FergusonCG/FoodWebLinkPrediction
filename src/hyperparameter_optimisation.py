import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.transforms import ToDevice
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler


from utils import (
    train,
    evaluate,
    build_model,
    ndcg_at_k,
    r2_score,
    mape,
)
from data import (
    EcologicalGraphLoader,
    combine_graphs,
)
from config import DEVICE

# Send all graphs directly to the correct device
_to_device = ToDevice(DEVICE)

# Seed torch for reporoducability
torch.manual_seed(42)

ENCODERS = {
    "gcn"        : {},
    "graphsage"  : {}, 
    "gatv2"      : {},
    "transformer": {}, 
    "simple"     : {},
}

class GNNHPO:
    """Hyper‑parameter search over encoder and decoder architectures."""

    def __init__(self):
        pass

    def train_one_model(self, model,  data, epochs, loss_fn):
        """
        Train `model on `data` for `epochs` and return loss observed in the final epoch.
        
        Parameters
        ----------
        model : models.LinkPredictionModel
            The model to be trained.
        data : torch_geometric.data.Data
            The graph data containing node features, edge indices, and edge weights.
        epochs : int
            The number of epochs to train the model.
        loss_fn : Callable
            The loss function to be used for training. For example, `torch.nn.L1Loss()`.

        Returns
        -------
        float
            The loss value from the last training epoch.
        """
        loss = 0.0
        for _ in range(epochs):
            loss = train(
                model,
                data,
                train_idx=torch.arange(data.edge_index.size(1)),
                optimizer=model.optimizer,
                loss_fn=loss_fn,
            )
        return float(loss)

    def hyperparameter_optimization_logo(
        self,
        encoder_type,
        dataset_paths,
        n_trials=50,
        epochs=50,
        log_path="hyperparam_logo_logs.csv",
        use_bin_adj=False,
        pca_components=0,
    ):
        """
        Run LOGO‑CV HPO and return the best Optuna parameter dictionary.
        
        Parameters
        ----------
        encoder_type : str
            The type of encoder architecture to be used. Options include "gcn", "graphsage", "gatv2", "transformer", and "simple".
        dataset_paths : List[Tuple[str, str]]
            A list of tuples, where each tuple contains the path to the JSON file for a food web's edge
            weight matrix and the corresponding node features file.
        n_trials : int, optional
            The number of Optuna trials to run. Default is 50.
        epochs : int, optional
            The number of epochs to train the model in each trial. Default is 50.
        log_path : str, optional
            The path to save the Optuna study results. Default is "hyperparam_logo_logs.csv".
        pca_components : int, optional
            The number of PCA components to use for dimensionality reduction. Default is 0 (no PCA).
        """
        food_webs = [
            _to_device(
                EcologicalGraphLoader(
                    edges_path,
                    node_features_path,
                    pca_components=pca_components,
                ).get_data()
            )
            for edges_path, node_features_path in dataset_paths
        ]
        loss_fn = nn.L1Loss()

        
        def objective(trial, pca_components=pca_components):
            """
            Optuna objective function to be minimised. 

            Defines the search space for hyper-parameters and evaluates the model using LOGO-CV.

            Parameters
            ----------
            trial : optuna.Trial
                The current Optuna trial object.
            pca_components : int
                The number of PCA components used (for logging). This has to be passed as a parameter as 
                PCA is applied outside of this function.

            Returns
            -------
            float
                The average MAE across all folds to be minimised.
            """
            # Encoder search space
            aggregator = None
            heads = 1
            dropout = 0.0
            
            if encoder_type == "transformer":
                heads = trial.suggest_int("transformer_heads", 1, 10)
                dropout = trial.suggest_float(
                    "transformer_dropout", 0.0, 0.5, step=0.1
                )
                # Limit size and depth of UniMP due to compute requirements
                n_encoder_layers = trial.suggest_int("n_encoder_layers", 1, 4)
                encoder_layer_widths = [
                    trial.suggest_int(f"encoder_layer_width_{i}", 16, 208, step=16)
                    for i in range(n_encoder_layers)
                ]
            else:
                if encoder_type == "graphsage":
                    aggregator = trial.suggest_categorical(
                        "graphsage_aggregator", ["mean", "max", "add"]
                    )
                elif encoder_type == "gatv2":
                    heads = trial.suggest_int("gatv2_heads", 1, 16)
                    dropout = trial.suggest_float("gatv2_dropout", 0.0, 0.5, step=0.1)
                
                # If simple encoder (MLP-only) then n_enc and enc_sizes are ignored so fill manually
                if encoder_type == "simple":
                    n_encoder_layers = 1
                    encoder_layer_widths=[1]
                else:
                    n_encoder_layers = trial.suggest_int("n_encoder_layers", 1, 5)
                    encoder_layer_widths = [
                        trial.suggest_int(f"encoder_layer_width_{i}", 16, 256, step=16)
                        for i in range(n_encoder_layers)
                    ]

            # Decoder search space
            n_decoder_layers = trial.suggest_int("n_decoder_layers", 1, 5)
            decoder_layer_widths = [
                trial.suggest_int(f"decoder_layer_width_{j}", 16, 256, step=16)
                for j in range(n_decoder_layers)
            ]

            lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

            fold_mae, fold_r2, fold_mape = [], [], []
            fold_ndcg10, fold_ndcg20, fold_ndcg50 = [], [], []

            for left_out_food_web_idx, val_data in enumerate(food_webs):
                train_data = combine_graphs(
                    [food_web for idx, food_web in enumerate(food_webs) if idx != left_out_food_web_idx]
                )

                model = build_model(
                    encoder_type=encoder_type,
                    aggregator=aggregator,
                    heads=heads,
                    dropout=dropout,
                    encoder_layer_sizes=encoder_layer_widths,
                    decoder_layer_sizes=decoder_layer_widths,
                    in_features=train_data.x.size(1),
                    lr=lr,
                )

                train_loss = self.train_one_model(
                    model, train_data, epochs, loss_fn
                )

                # Prune trial if training loss is too high
                trial.report(train_loss, left_out_food_web_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if use_bin_adj:
                    eval_graph = Data(
                        x=val_data.x,
                        edge_index=val_data.edge_index,
                        edge_weights=torch.ones_like(val_data.edge_weights),
                    ).to(DEVICE)
                else:
                    eval_graph = val_data 

                val_mae, preds = evaluate(
                    model,
                    eval_graph,
                    val_data.edge_index,
                    val_data.edge_weights,
                    loss_fn,
                )

                fold_mae.append(val_mae)
                fold_r2.append(r2_score(preds, val_data.edge_weights))
                fold_mape.append(mape(preds, val_data.edge_weights))
                fold_ndcg10.append(ndcg_at_k(preds, val_data.edge_weights, 10))
                fold_ndcg20.append(ndcg_at_k(preds, val_data.edge_weights, 20))
                fold_ndcg50.append(ndcg_at_k(preds, val_data.edge_weights, 50))

                # Save metrics as custom attributes
                trial.set_user_attr(f"fold{left_out_food_web_idx+1}_mae",  val_mae)
                trial.set_user_attr(f"fold{left_out_food_web_idx+1}_r2",   fold_r2[-1])
                trial.set_user_attr(f"fold{left_out_food_web_idx+1}_mape", fold_mape[-1])
                trial.set_user_attr(f"fold{left_out_food_web_idx+1}_ndcg10",  fold_ndcg10[-1])
                trial.set_user_attr(f"fold{left_out_food_web_idx+1}_ndcg20",  fold_ndcg20[-1])
                trial.set_user_attr(f"fold{left_out_food_web_idx+1}_ndcg50",  fold_ndcg50[-1])

            # Save average metrics across all folds as custom attributes
            trial.set_user_attr("avg_mae",  float(np.mean(fold_mae)))
            trial.set_user_attr("avg_r2",   float(np.mean(fold_r2)))
            trial.set_user_attr("avg_mape", float(np.mean(fold_mape)))
            trial.set_user_attr("avg_ndcg10",  float(np.mean(fold_ndcg10)))
            trial.set_user_attr("avg_ndcg20",  float(np.mean(fold_ndcg20)))
            trial.set_user_attr("avg_ndcg50",  float(np.mean(fold_ndcg50)))

            trial.set_user_attr("pca_components", pca_components)

            # Return the average MAE across all folds to be minimised
            return float(np.mean(fold_mae))

        # Initiate Optuna study
        study = optuna.create_study(
            direction="minimize",
            pruner=SuccessiveHalvingPruner(),
            sampler=TPESampler(),
        )
        study.optimize(objective, n_trials=n_trials)

        # Save HPO results to CSV
        study.trials_dataframe().to_csv(log_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        return study.best_params

def main():
    """
    Command-line interface for running hyperparameter optimisation for each of the models defined in `ENCODERS`
    
    Arguments
    ---------
    food_webs_path : str
        Path to folder containing JSON food web edge weight matrices.
    node_features_path : str
        Path to folder containing JSON node feature matrices.
    n_trials : int
        Number of Optuna trials to run for each encoder. Default is 100.
    epochs : int
        Number of epochs to train each model. Default is 100.
    pca_components : int
        Number of PCA components to use for dimensionality reduction. Default is 0 (no PCA).
    out_dir : str
        Directory to save the HPO results. Default is "hpo_results".
    """
    p = argparse.ArgumentParser(
        description="Run separate HPO studies per encoder architecture."
    )
    p.add_argument("--food_webs_path", type=str,
                   default="data_processing/final_foodwebs",
                   help="Folder with food‑web adjacency JSONs.")
    p.add_argument("--node_features_path", type=str,
                   default="data_processing/final_node_features",
                   help="Folder with species‑feature JSONs.")
    p.add_argument("--n_trials", type=int, default=100,
                   help="Optuna trials per architecture (default: 100).")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs per trial (default: 100).")
    p.add_argument("--pca_components", type=int, default=0,
                   help="Number of PCA components to use (default: 0).")
    p.add_argument("--out_dir", type=str, default="hpo_results",
                   help="Directory to save HPO results (default: hpo_results).")
    
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset_paths = list(zip(
        sorted(
            os.path.join(args.food_webs_path, f)
            for f in os.listdir(args.food_webs_path) if f.endswith(".json")
        ),
        sorted(
            os.path.join(args.node_features_path, f)
            for f in os.listdir(args.node_features_path) if f.endswith(".json")
        ),
    ))

    hpo = GNNHPO()

    best_configs = []
    for encoder in ENCODERS:
        print("\n═════════════════════════════════════════════════════════════════════")
        print(f"Searching hyper‑parameters for encoder: {encoder.upper()}")
        print("═════════════════════════════════════════════════════════════════════")

        best = hpo.hyperparameter_optimization_logo(
            dataset_paths=dataset_paths,
            encoder_type=encoder, 
            n_trials=args.n_trials,
            epochs=args.epochs,
            log_path=os.path.join(args.out_dir, f"hyperparam_logo_logs_{encoder}.csv"),
            pca_components=args.pca_components,
        )

        print(f"\n⟡ Best hyper‑parameters for {encoder}:\n{best}\n")

        best["encoder_type"] = encoder
        best["pca_components"] = args.pca_components
        best_configs.append(best)
    
    with open(os.path.join(args.out_dir, "hpo_best_configs.json"), "w") as f:
        json.dump(best_configs, f)

if __name__ == "__main__":
    main()


