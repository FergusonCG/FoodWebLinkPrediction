import argparse
import json
import pandas as pd
import re
import os
import numpy as np


def join_duplicate_species(json_data):
    """
    Remove duplication in The Web of Life's Species data by joining species with the same name.

    Because The Web of Life create two separate entries for the same species if they act as both a predator and prey,
        the information is joined into a single row for ease of use/reading.

    Parameters
    ----------
    json_data : dict
        Output JSON file from `import_node_features.R` containing taxonomic node features.

    Returns
    -------
    dict
        Dictionary object with each row representing unique species and their features.
    """
    joined_species = {}
    for idx, species_info in json_data["node features"].items():        
        # Remove "kingdom" as its a duplicate feature created during taxonomic lookup and less reliable than "Kingdom" from Web of Life
        species_info.pop("kingdom", None)

        if species_info["species_name"] in joined_species.keys():
            duplicate_species = joined_species[species_info["species_name"]].copy()
            for trait, value in species_info.items():
                # Don't modify species names, and dont include "NA" feature column returned from taxonomic databases.
                if not re.match("(^specie*|^na)", str.lower(trait)):
                    joined_species[species_info["species_name"]][str.lower(trait)] = list(set([duplicate_species[str.lower(trait)], value]))
        else:
            joined_species[species_info["species_name"]] = {str.lower(trait) : value for trait, value in species_info.items() if not re.match("(^specie*|^na)", str.lower(trait))}

    return joined_species

def is_missing(x):
    """
    Return true if the value is missing, else return False.

    Used to identify missing values in `get_high_coverage_columns`.

    Parameters
    ----------
    x : any
        Value to check for missingness.
    
    Returns
    -------
    bool
        True if the value is missing, else False.
    """
    return (
        x is None or
        (isinstance(x, float) and pd.isna(x)) or 
        x == "" or
        (isinstance(x, list) and len(x) == 1 and x[0] is None)
    )

def get_high_coverage_columns(json_node_features_dir="json_node_features"):
    """
    Get columns with mean coverage of over 50% across all food webs.

    Many taxonomic features are missing for over 99% of nodes across food webs, therefore we use only those
        with >50% mean coverage to minimise the impact poor quality data has on predictions.

    Parameters
    ----------
    json_node_features_dir : str
        Path to folder containing JSON node features as output by `import_node_features.R`

    Returns
    -------
    dict
        Dictionary where keys are the names of features that are missing for less than 50% of nodes across all
        food webs in `json_node_features_dir`, and values are the corresponding coverage for that feature.
    """
    coverages = {}
    for node_features_file in os.listdir(json_node_features_dir):
        with open(f"{json_node_features_dir}/{node_features_file}", "r") as f:
            json_data = json.load(f)
        df = pd.DataFrame.from_dict(join_duplicate_species(json_data), orient="index")

        # Drop columns that describe the dependent variable
        df = df.drop(["degree", "sum.connections.strength.", "networks.presence"], axis=1)

        # Calculate coverage of each feature per food web
        feature_coverage = (len(df) - df.map(is_missing).sum())/len(df)
        for feature in feature_coverage.index:
            if feature not in coverages.keys():
                coverages[feature] = []
            coverages[feature].append(feature_coverage[feature])

    # Calculate mean coverage of each feature across all food webs
    mean_coverage = {feature : np.mean(coverage) for feature, coverage in coverages.items()}
    return {feature : coverage for feature, coverage in mean_coverage.items() if coverage > 0.5}

def one_hot_encode(
        json_node_features_dir="json_node_features", 
        features_to_keep=["kingdom", "role", "phylum", "order", "family", "genus", "class"],
        output_dir="ohe_node_features"
    ):
    """
    One Hot Enocode taxonomic features for each food web as they are categorical.

    Joining duplicate species in `join_duplicate_species` allows for multi-label encoding of features.
        E.g. if a species is both a predator and prey, it will be encoded as both a predator and prey
        with a `1` in the respective columns for `Predator` and `Prey`.

    Parameters
    ----------
    json_node_features_dir : str
        Directory containing JSON files with node features for each food web produced by `import_node_features.R`.
    features_to_keep : list
        List of features to keep in the final one-hot encoded dataframe. Default value is features identified as
        having >50% mean coverage across all food webs.
    output_dir : str
        Directory to save the one-hot encoded dataframes to.

    Returns
    -------
    None
        The function saves the one-hot encoded dataframes to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for node_features_file in os.listdir(json_node_features_dir):
        with open(f"{json_node_features_dir}/{node_features_file}", "r") as f:
            json_data = json.load(f)
        df = pd.DataFrame.from_dict(join_duplicate_species(json_data), orient="index")

        # Drop columns that describe the dependent variable
        df = df.drop(["degree", "sum.connections.strength.", "networks.presence"], axis=1)

        # Drop all features not in `features_to_keep`
        df = df[features_to_keep]

        for col in df.columns:
            exploded = df[[col]].explode(col)
            dummies = pd.get_dummies(exploded[col], prefix=col)
            dummies = dummies.groupby(exploded.index).sum()
            df = df.drop(col, axis=1).join(dummies)

        df.to_json(f"{output_dir}/{node_features_file}", orient="index")

        print(f"✅  Wrote {output_dir}/{node_features_file}")

def standardise_ohe_columns(
        ohe_node_features_dir="ohe_node_features", 
        output_dir="final_node_features"
    ):
    """
    Standardise columns across all food webs to ensure they are the same dimension.

    As models expect the same feature dimensions across food webs, standardise by collecting all
        column names produced in `one_hot_encode()`, adding any missing columns to each dataframe,
        and filling column values with 0 if the column did not exist previously.

    Parameters
    ----------
    ohe_node_features_dir : str
        Directory containing JSON files with one-hot encoded node features for each food web produced by `one_hot_encode()`.
    output_dir : str
        Directory to save the standardised one-hot encoded dataframes to.

    Returns
    -------
    None
        The function saves the standardised one-hot encoded dataframes to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_cols = []
    for ohe_features_file in os.listdir(ohe_node_features_dir):
        df = pd.read_json(f"{ohe_node_features_dir}/{ohe_features_file}", orient="index")
        all_cols += df.columns.to_list()
    all_cols = list(set(all_cols))
    print(all_cols)

    for ohe_features_file in os.listdir(ohe_node_features_dir):
        df = pd.read_json(f"{ohe_node_features_dir}/{ohe_features_file}", orient="index")

        species_in_df = df.index.copy()
        df = df.reindex(
            index=species_in_df,
            columns=all_cols, 
            fill_value=0
        )
        
        df.rename_axis("species_name").reset_index().to_json(f"{output_dir}/{ohe_features_file}", orient="index")

        print(f"✅  Wrote {output_dir}/{ohe_features_file}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode node features into ML-acceptable format and save to local machine"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="json_node_features",
        help="Path to folder containing input .json node feature files"
    )
    parser.add_argument(
        "--ohe_dir",
        type=str,
        default="ohe_node_features",
        help="Path to folder to output non-standardised one-hot encoded node feature files"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="final_node_features",
        help="Output folder for final node features"
    )
    args = parser.parse_args()

    print("** Creating one-hot encodings per food web **")
    one_hot_encode(
        json_node_features_dir=args.input_dir, 
        features_to_keep=list(get_high_coverage_columns(json_node_features_dir="json_node_features").keys()),
        output_dir=args.ohe_dir
    )

    print("** Creating standardise ohe-hot encodings across food webs **")
    standardise_ohe_columns(
        ohe_node_features_dir=args.ohe_dir, 
        output_dir=args.out_dir
    )