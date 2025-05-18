from torch_geometric.data import Data
import torch
import json
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA


def combine_graphs(graphs):
    """
    Combine edges from multiple graphs into a single data source.

    Instead of training on single graphs and updating the weights accordingly, this research uses 
    Leave-One-Group-Out (LOGO) cross-validation to train on all graphs except one. This function is used
    to combine all graphs except the one left out into a single graph, with node indicies shifted to ensure
    edges remain valid.

    Parameters
    ----------
    graphs : List[Data]
        A list of food web graphs, with each food web represented as a PyTorch Geometric Data object.
    """
    if len(graphs) == 1:
        return graphs[0]

    node_features, edge_idxs, edge_weights = [], [], []
    node_offset = 0
    for g in graphs:
        node_features.append(g.x)
        edge_idxs.append(g.edge_index + node_offset)
        edge_weights.append(g.edge_weights)
        node_offset += g.num_nodes
    x_all = torch.cat(node_features, dim=0)
    edge_index_all = torch.cat(edge_idxs, dim=1)
    edge_w_all = torch.cat(edge_weights, dim=0)

    return Data(x=x_all, edge_index=edge_index_all, edge_weights=edge_w_all)


class EcologicalGraphLoader:
    """
    Wrapper for loading a single ecological network and associated node features from JSON files.


    Loads node features and edge data from JSON files, processes the node features
    (with one-hot encoding for selected categorical columns), and constructs a 
    PyTorch Geometric Data object.
    """
    def __init__(
            self, 
            edges_path, 
            node_features_path, 
            pca_components=0,
            pca_whiten=False,
            pca_random_state=42
        ):
        """
        Instantiate the EcologicalGraphLoader.

        Parameters
        ----------
        edges_path : str
            Path to the JSON file containing the edge weight matrix.
        node_features_path : str
            Path to the JSON file containing node features.
        pca_components : int, optional
            Number of PCA components to keep. If 0, no PCA is applied. Default is 0.
        pca_whiten : bool, optional
            Whether to whiten the PCA components. Default is False.
        pca_random_state : int, optional
            Random state for PCA reproducibility. Default is 42.
        """
        self.edges_path = edges_path
        self.node_features_path = node_features_path
        # If not provided, use a default set of taxonomic features.
        self.pca_components = pca_components
        self.pca_whiten = pca_whiten
        self.pca_random_state = pca_random_state
        self.data = None
    

    def load_node_features(self):
        """
        Load node features from the `node_features_path` JSON file specified during instantiation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the node features, with species names as the index.
        """

        df = pd.read_json(self.node_features_path, orient="index")
        # Create a mapping from species name to index
        self.species_to_index = {name: idx for idx, name in enumerate(df["species_name"])} 

        df_features_only = df.drop("species_name", axis=1)
        features_tensor = torch.tensor(df_features_only.values, dtype=torch.float32)
        return features_tensor
    

    def load_edges(self):
        """
        Load edges from the `edges_path JSON file specified during instantiation.

        Returns
        -------
        dict
            A dictionary where keys are source species names, and values are dictionaries of target species names
            and the corresponding edge weight. E.g. {source_species: {target_species: weight, ...}}.
        """
        with open(self.edges_path, 'r') as f:
            edges_data = json.load(f)
        return edges_data
    
    
    def process_edges(self):
        """
        Convert edge weight matrix into PyG-accepted format.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
                - edge_index: A tensor of shape [2, num_edges] representing the edge indices.
                - edge_weights: A tensor of shape [num_edges] representing the edge weights.
        """
        edges_data = self.load_edges()
        # Ensure species_to_index is set
        if not hasattr(self, "species_to_index"):
            df = self.load_node_features()
            self.species_to_index = {name: idx for idx, name in enumerate(df["species_name"])}

        src_list, dst_list, weight_list = [], [], []
        for src_species, targets in edges_data.items():
            for dst_species, weight in targets.items():
                if weight > 0:
                    if src_species in self.species_to_index and dst_species in self.species_to_index:
                        src_idx = self.species_to_index[src_species]
                        dst_idx = self.species_to_index[dst_species]
                        src_list.append(src_idx)
                        dst_list.append(dst_idx)
                        weight_list.append(weight)
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weights = torch.tensor(weight_list, dtype=torch.float)
        return edge_index, edge_weights
    

    def get_data(self):
        """
        Main loading function.

        Retrieves edge weight matrix and node features for a specified node features for a given
            Ecological Network.

        Returns
        -------
        Data
            A PyTorch Geometric Data object containing the processed node features and edge weights.
        """
        x = self.load_node_features()
        edge_index, edge_weights = self.process_edges()
        # If pca_components is 0, skip; otherwise PCA() returns empty feature set
        if self.pca_components > 0:
            # move to CPU / numpy for sklearn
            x_np = x.detach().cpu().numpy()
            pca = PCA(
                n_components=self.pca_components,
                whiten=self.pca_whiten,
                random_state=self.pca_random_state,
            )
            x_reduced = pca.fit_transform(x_np)
            x = torch.from_numpy(x_reduced).to(torch.float32)
        self.data = Data(x=x, edge_index=edge_index, edge_weights=edge_weights)
        return self.data
