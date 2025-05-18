import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import torch
from torch_geometric.data import Data
from typing import Dict, List, Any

from config import DEVICE
from data import EcologicalGraphLoader, combine_graphs
from utils import train, evaluate, ndcg_at_k, mape, r2_score, model_from_params


def get_mean_from_string_result(string_metric):
    """
    Return the mean component from a "x ± y" string – stripped of % if MAPE is used.
    
    Parameters
    ----------
    string_metric : str
        Metric string, for example: "0.1234 ± 0.5678" or "12.34% ± 5.67%".

    Returns
    -------
    float
        The mean component of the metric string, converted to a float.
    """
    return float(string_metric.split("±")[0].replace("%", "").strip())


def run_logo(
    dataset_paths,
    model_configuration,
    epochs=50,
    edge_dropout=0.0,
    pca_components=0,
    use_bin_adj=False,
    dump_preds=False,
    out_dir="logo_preds",
):
    """
    Train and evaluate a given model architecture on the given datasets, saving and plotting the results
        for use in further evaluations.

    Parameters
    ----------
    dataset_paths : List[Tuple[str, str]] 
        A list of tuples, where each tuple contains the path to the JSON file for node features in a given
        food web and the corresponding food web edge weight matrix.
    model_configuration : Dict
        Configuration for a single model instance. Though intended for model configurations as found during
        HPO, custom configurations can be used by passing a dictionary of the same structure.
    epochs : int
        Number of epochs to train each model.
    edge_dropout : float
        Probability for dropping each edge prior to message passing.
    pca_components : int
        Number of PCA components to use for dimensionality reduction. If 0, no PCA is applied.
    use_bin_adj : bool
        Flag to apply binary edge weight masks to all edges in the test set input.
    dump_preds : bool
        Flag to save the final predictions of models to machine.
    out_dir : Path | str
        Path to folder where the results will be saved. If it does not exist, it will be created.
    """
    food_webs = [
        EcologicalGraphLoader(e, n, pca_components=pca_components).get_data().to(DEVICE)
        for e, n in dataset_paths
    ]
    loss_fn = torch.nn.L1Loss()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {metric: [] for metric in ("MAE", "R2", "MAPE", "ndcg10", "ndcg20", "ndcg50")}
    all_train_curves: List[List[float]] = []
    all_val_curves: List[List[float]] = []

    for fold_idx, test_web in enumerate(food_webs):
        train_combined_web = combine_graphs([food_web for idx, food_web in enumerate(food_webs) if idx != fold_idx])

        # ----- split val edges 60/40 ---------------------------------------
        num_e = test_web.edge_index.size(1)
        perm = torch.randperm(num_e)
        cut = int(0.6 * num_e)
        test_input_idx, test_pred_idx = perm[:cut], perm[cut:]

        test_input_graph = Data(
            x=test_web.x,
            edge_index=test_web.edge_index[:, test_input_idx],
            edge_weights=test_web.edge_weights[test_input_idx],
        ).to(DEVICE)
        test_pred_edges = test_web.edge_index[:, test_pred_idx]
        test_pred_weights = test_web.edge_weights[test_pred_idx]

        model = model_from_params(model_configuration, in_features=train_combined_web.x.size(1))
        train_idx = torch.arange(train_combined_web.edge_index.size(1), device=DEVICE)

        train_losses: List[float] = []
        val_losses: List[float] = []

        # ----- epoch loop ---------------------------------------------------
        for ep in range(1, epochs + 1):
            tr_loss = train(
                model=model,
                food_web_graph=train_combined_web,
                train_idx=train_idx,
                optimizer=model.optimizer,
                loss_fn=model.loss_fn,
                edge_dropout=edge_dropout,
            )
            train_losses.append(tr_loss)

            # validation loss (on training adjacency of val web)
            v_mae, _ = evaluate(model, test_input_graph, test_pred_edges, test_pred_weights, loss_fn)
            val_losses.append(v_mae)

        # ----- evaluation adjacency selection ------------------------------
        if use_bin_adj:
            test_input_graph = Data(
                x=test_web.x,
                edge_index=test_web.edge_index,
                edge_weights=torch.ones_like(test_web.edge_weights),
            ).to(DEVICE)
        else:
            test_input_graph = test_input_graph

        mae, preds = evaluate(model, test_input_graph, test_pred_edges, test_pred_weights, loss_fn)

        metrics["MAE"].append(mae)
        metrics["R2"].append(r2_score(preds, test_pred_weights))
        metrics["MAPE"].append(mape(preds, test_pred_weights))
        metrics["ndcg10"].append(ndcg_at_k(preds, test_pred_weights, 10))
        metrics["ndcg20"].append(ndcg_at_k(preds, test_pred_weights, 20))
        metrics["ndcg50"].append(ndcg_at_k(preds, test_pred_weights, 50))

        print(
            f"Fold {fold_idx+1}/{len(food_webs)}  "
            f"MAE {mae:.4f}   "
            f"R² {metrics['R2'][-1]:.4f}   "
            f"MAPE {metrics['MAPE'][-1]:.2f}%   "
            f"ndcg10 {metrics['ndcg10'][-1]:.3f}   "
            f"ndcg20 {metrics['ndcg20'][-1]:.3f}   "
            f"ndcg50 {metrics['ndcg50'][-1]:.3f}"
        )

        # Save learning curver per fold
        epoch_range = list(range(1, epochs + 1))
        plt.figure()
        plt.plot(epoch_range, train_losses, label="train MAE")
        plt.plot(epoch_range, val_losses, label="val MAE")
        plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title(f"Learning Curve – Fold {fold_idx+1}")
        plt.grid(True); plt.legend()
        plt.savefig(out_dir / f"{model_configuration['encoder_type']}_curves_fold_{fold_idx+1:02d}.png")
        plt.close()

        all_train_curves.append(train_losses)
        all_val_curves.append(val_losses)

        # Save predictions if flag is True
        if dump_preds:
            with open(out_dir / f"{model_configuration['encoder_type']}_fold{fold_idx+1:02d}_preds.json", "w") as f:
                json.dump(
                    {
                        "edge_index": test_pred_edges.cpu().tolist(),
                        "true_wts": test_pred_weights.cpu().tolist(),
                        "pred_wts": preds.cpu().tolist(),
                    },
                    f
                )

    # Save averaged learning curves across folds
    train_mat = torch.tensor(all_train_curves)
    val_mat = torch.tensor(all_val_curves)
    ep = torch.arange(1, epochs + 1)
    plt.figure()
    plt.plot(ep, train_mat.mean(0), label="train mean")
    plt.fill_between(ep, train_mat.mean(0)-train_mat.std(0), train_mat.mean(0)+train_mat.std(0), alpha=0.2)
    plt.plot(ep, val_mat.mean(0), label="val mean")
    plt.fill_between(ep, val_mat.mean(0)-val_mat.std(0), val_mat.mean(0)+val_mat.std(0), alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title("Aggregate Learning Curves (mean ± SD)")
    plt.grid(True); plt.legend()
    plt.savefig(out_dir / f"{model_configuration['encoder_type']}_aggregate_learning_curves.png")
    plt.close()


    train_curve_mean = train_mat.mean(0).tolist()
    val_curve_mean = val_mat.mean(0).tolist()

    def _avg(values, percentage=False):
        """
        Helper function to calculate the mean and standard deviation of a list and return as readable string format.

        Parameters  
        ----------
        values : List[float]
            List of values to calculate mean and std for.
        percentage : bool
            Flag to indicate if the values are percentages. If True, format as percentage string with fewer decimal points.

        Returns
        -------
        str
            Formatted string with mean and std values. E.g. "0.1234 ± 0.5678" or "12.34% ± 5.67%" for MAPE.
        """
        tensor_values = torch.tensor(values)
        mean = tensor_values.mean().item()
        std = tensor_values.std().item()
        return f"{mean:.2f}% ± {std:.2f}%" if percentage else f"{mean:.4f} ± {std:.4f}"

    # Return performance metrics and average training curves (used to plot average across runs)
    return {
        "enc": model_configuration["encoder_type"],
        "MAE": _avg(metrics["MAE"]),
        "R2": _avg(metrics["R2"]),
        "MAPE": _avg(metrics["MAPE"], percentage=True),
        "ndcg10": _avg(metrics["ndcg10"]),
        "ndcg20": _avg(metrics["ndcg20"]),
        "ndcg50": _avg(metrics["ndcg50"]),
        "train_curve": train_curve_mean,
        "val_curve": val_curve_mean,
    }


def main():
    """
    Command-line interface for performing the final train/val/test over the best
        configurations found during HPO.

    This script repeats the train/val/test pipeline multiple times for each model 
        configuration within the `best_params` file and averages the results to mitigate
        any bias introduced by a single run.

    Arguments
    ---------
    best_params : str
        Path to JSON file containing the best hyperparameter configurations.
    webs_path : str
        Path to folder containing JSON food web edge weight matrices.
    species_path : str
        Path to folder containing JSON node feature matrices.
    epochs : int
        Number of epochs to train each model.
    n_runs : int
        Number of times to repeat the training for each encoder.
    use_bin_adj : bool
        Flag to apply binary edge weight masks to all edges in the test set input.
    dump_preds : bool
        Flag to save the final predictions of models to machine.
    out_dir : str
        Path to folder where the results will be saved.
    """
    p = argparse.ArgumentParser(
        description="Run each encoder config N times and average the metrics."
    )

    p.add_argument("--best_params", type=str, required=True,
                   help="JSON file with *one or many* best hyperparameter dicts.")
    p.add_argument("--webs_path", type=str, default="data_processing/final_foodwebs",
                   help="Folder containing food‑web edge JSONs (same as original script).")
    p.add_argument("--species_path", type=str, default="data_processing/final_node_features",
                   help="Folder with node‑feature JSONs (same order as webs_path).")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs per run.")
    p.add_argument("--n_runs", type=int, default=100, help="How many repetitions per encoder.")
    p.add_argument("--pca_components", type=int, default=0,
                   help="Number of PCA components to use for dimensionality reduction.")
    p.add_argument("--use_bin_adj", action="store_true")    
    p.add_argument("--dump_preds", action="store_true")
    p.add_argument("--out_dir", type=str, default="multi_run_results",
                   help="Top‑level folder collecting all encoder sub‑folders.")

    args = p.parse_args()
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load model configurations
    with open(args.best_params) as f:
        configs: Any = json.load(f)
    # If single config in file, convert to list for consistency
    if isinstance(configs, dict):
        configs = [configs]

    # Create List[Tuple(food_web_path, node_features_path)] format, as used in EcologicalGraphLoader
    edges = sorted(Path(args.webs_path).glob("*.json"))
    feats = sorted(Path(args.species_path).glob("*.json"))
    dataset_paths: List[tuple[str, str]] = [(str(e), str(f)) for e, f in zip(edges, feats)]

    # Main process
    metrics = ["MAE", "R2", "MAPE", "ndcg10", "ndcg20", "ndcg50"]
    aggregated_results = {}
    train_curves = {}
    val_curves = {}
    # Loop over each encoder type
    for config_idx, config in enumerate(configs):
        encoder_type = config["encoder_type"]
        aggregated_results[encoder_type] = {m: [] for m in metrics}
        train_curves[encoder_type] = []
        val_curves[encoder_type] = []
        enc_out_dir = Path(args.out_dir) / encoder_type

        print(f"\n================ {encoder_type} ================")

        # As SimpleEncoder HPO configs don't have encode layer/width params, manually fill
        if config["encoder_type"] == "simple":
            config.setdefault("n_encoder_layers", 1)
            config.setdefault("encoder_layer_width_0", 1)

        # Loop over each run - training and evaluating the model across each
        for run_idx in range(1, args.n_runs + 1):
            print(f"\n──── run {run_idx}/{args.n_runs} ────")
            torch.manual_seed(1000 * (config_idx + 1) + run_idx)

            results = run_logo(
                dataset_paths=dataset_paths,
                model_configuration=config,
                epochs=args.epochs,
                pca_components=config["pca_components"],
                use_bin_adj=args.use_bin_adj,
                dump_preds=args.dump_preds,
                out_dir=enc_out_dir / f"run_{run_idx:02d}",
            )

            for metric in metrics:
                aggregated_results[encoder_type][metric].append(get_mean_from_string_result(results[metric]))

            # Collect learning curves from run_logo() output
            train_curves[encoder_type].append(results["train_curve"])
            val_curves[encoder_type].append(results["val_curve"])

        # Save averaged learning curves across runs
        train_curve_matrix = torch.tensor(train_curves[encoder_type]) 
        val_curve_matrix   = torch.tensor(val_curves[encoder_type])
        plot_epochs = torch.arange(1, args.epochs + 1)
        plt.figure()
        plt.plot(plot_epochs, train_curve_matrix.mean(0), label="train mean")
        plt.fill_between(plot_epochs,
                         train_curve_matrix.mean(0) - train_curve_matrix.std(0),
                         train_curve_matrix.mean(0) + train_curve_matrix.std(0), alpha=0.2)
        plt.plot(plot_epochs, val_curve_matrix.mean(0), label="val mean")
        plt.fill_between(plot_epochs,
                         val_curve_matrix.mean(0) - val_curve_matrix.std(0),
                         val_curve_matrix.mean(0) + val_curve_matrix.std(0), alpha=0.2)
        plt.xlabel("Epoch"); plt.ylabel("MAE")
        plt.title(f"Learning Curves across {args.n_runs} runs – {encoder_type}")
        plt.grid(True); plt.legend()
        plt.tight_layout()
        plt.savefig(enc_out_dir / "learning_curves_all_runs.png")
        plt.close()

    # Save and print final results
    summary: Dict[str, Dict[str, str]] = {}
    print("\n================ AVERAGED METRICS ================")
    for encoder_type, encoder_metrics in aggregated_results.items():
        print(f"\n*** {encoder_type} ***")
        summary[encoder_type] = {}
        for metric, values in encoder_metrics.items():
            tensor_values = torch.tensor(values)
            mean = tensor_values.mean().item()
            std = tensor_values.std().item()
            # Use fewer decimal places for MAPE
            result = f"{mean:.2f}% ± {std:.2f}%" if metric == "MAPE" else f"{mean:.4f} ± {std:.4f}"
            summary[encoder_type][metric] = result
            print(f"{metric}: {result}  (n_runs={args.n_runs})")

    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save box-plots of final results
    print("\n================ Saving box‑plots ================")
    plot_dir = output_root / "plots"
    plot_dir.mkdir(exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(max(6, len(aggregated_results) * 1.2), 6))
        data = [encoder_metrics[metric] for encoder_metrics in aggregated_results.values()]
        labels = list(aggregated_results.keys())
        boxplot = plt.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=True,
            meanprops=dict(color="blue", linewidth=1),
            medianprops=dict(color="orange", linewidth=1),
        )

        # legend elements ------------------------------------------------
        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=2, label="Mean"),
            Line2D([0], [0], color="orange", linewidth=2, label="Median"),
        ]
        plt.legend(handles=legend_elements, loc="best")
        plt.ylabel(metric)
        plt.title(f"{metric} distribution across {args.n_runs} runs")
        plt.xticks(rotation=45, ha="right")
        plt.legend(loc="best")
        plt.tight_layout()
        file_name = plot_dir / f"{metric}_boxplot.png"
        plt.savefig(file_name)
        plt.close()
        print(f"⤷  {metric} → {file_name}")


if __name__ == "__main__":
    main()
