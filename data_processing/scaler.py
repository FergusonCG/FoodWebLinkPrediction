import os
import json
import pandas as pd
from pathlib import Path
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_csv_food_webs(dir):
    """
    Process CSV food webs into a square matrix and convert to JSON.

    Parameters
    ----------
    dir : str
        Directory containing the CSV food webs to process. The specified directory
        may contain non-CSV files however should not include CSVs that are not food webs.
    
    Returns
    -------
    None
        The function saves the processed food webs as JSON files in the same directory.
    """
    for file in os.listdir(dir):
        csv_path  = Path(file) 
        if str(csv_path).split(".")[-1] != "csv":
            continue
        df = (pd.read_csv(f"{dir}/{csv_path}", index_col=0, keep_default_na=False)
                .apply(pd.to_numeric, errors="coerce"))   # forces numeric dtype

        # Because some rows do not appear as columns and vice versa, first create a 
        #   square matrix with all spceies in the rows and columns.
        all_node_names = sorted(set(df.index) | set(df.columns))
        df = df.reindex(index=all_node_names, columns=all_node_names, fill_value=0.0)

        # Convert square matrix to a nested dictionary 
        nested = {
            source_node: {target_node: float(edge_weight) for target_node, edge_weight in row.items()}
            for source_node, row in df.to_dict(orient="index").items()
        }

        # Write to JSON
        json_path = Path(f"{dir}/{file.split(".")[0]}.json")
        with json_path.open("w") as f:
            json.dump(nested, f, indent=2)

        print(f"Saved to {json_path}")


class NetworkScaler:
    """
    Scales each food-web JSON’s edge weights independently via MinMax scaling.
    """
    def __init__(self):
        """
        Instantiate NetworkScaler class.
        """
        self._edges_cache = {}

    def _load_adj(self, path):
        """
        Load the weighted adjacency matrix from a JSON file and cache in class
        attributes.

        Parameters
        ----------
        path : str
            Path to the JSON file containing the weighted adjacency matrix.
        """
        if path not in self._edges_cache:
            with open(path, 'r') as f:
                self._edges_cache[path] = json.load(f)
        return self._edges_cache[path]

    def fit_and_save(self, dataset_dir, out_dir="final_foodwebs"):
        """
        Scale each food-web JSON’s edge weights independently via MinMax scaling.

        Parameters
        ----------
        dataset_dir : str
            Path to folder containing input .json network files produced by `process_csv_food_webs()`.
        out_dir : str
            Output folder for the scaled JSONs. Default is "final_foodwebs".
        """
        os.makedirs(out_dir, exist_ok=True)

        # Collect all JSON files in `dataset_dir`
        json_files = sorted(
            f for f in os.listdir(dataset_dir)
            if f.lower().endswith('.json')
        )
        if not json_files:
            raise RuntimeError(f"No .json files found in {dataset_dir!r}")

        # Get full path for each JSON file found
        for fname in json_files:
            path = os.path.join(dataset_dir, fname)
            adj = self._load_adj(path)

            # Flatten edge weihts to 1D array so MinMaxScaler can be applied
            weights = np.array(
                [float(w) for targets in adj.values() for w in targets.values()],
                dtype=np.float32
            ).reshape(-1, 1)

            # Fit scaler and transform current food web into range(0,1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(weights).flatten()

            # Overwrite un-scaled edge weights in adjacency matrix
            idx = 0
            for src, targets in adj.items():
                for dst in list(targets):
                    targets[dst] = float(scaled[idx])
                    idx += 1

            # Save result
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{base}_scaled.json")
            with open(out_path, 'w') as f:
                json.dump(adj, f, indent=2)

        print(f"Individually-scaled networks saved to: {out_dir!r}")


def main():
    """
    Command-line interface for scaling food-web JSON files in a given folder.

    Arguments
    ---------
    dataset_dir : str
        Path to folder containing .csv food web files to scale.
    out_dir : str
        Output folder for the scaled JSONs. Default is "final_foodwebs".    
    """
    parser = argparse.ArgumentParser(
        description="Scale each food-web JSON independently with MinMaxScaler"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to folder containing input .csv food web files"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="final_foodwebs",
        help="Output folder for the scaled JSONs"
    )
    args = parser.parse_args()

    process_csv_food_webs(args.dataset_dir)

    scaler = NetworkScaler()
    scaler.fit_and_save(args.dataset_dir, args.out_dir)


if __name__ == "__main__":
    main()
