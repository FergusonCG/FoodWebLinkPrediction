import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, TransformerConv


class GCNEncoder(nn.Module):
    """
    Class to create encoder architectures comprised of Graph Convolutional Network layers.
    """
    def __init__(self, in_channels, layer_sizes):
        """
        Create/initialise dGCNEncoder object of custom depth.

        Parameters
        ----------
        in_channels : int
            Initial feature vector dimensions. E.g. if nodes contain 10 features, in_channels should be set to 10.
            Alternatively, in_channels can be set to -1 to infer the number of node features dynamically.
        layer_sizes : List[int]
            Output dimension of each GCN layer, where each integer element in layer_sizes is the output dimension of the i-th layer.
            E.g. layer_sizes = [64,32,16] would create three GCN layers with output dimensions 64, 32, and 16 respectively.

        Returns
        -------
        None        
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        # Start from in_channels for the first layer
        current_dim = in_channels
        for hidden_dim in layer_sizes:
            self.conv_layers.append(GCNConv(current_dim, hidden_dim))
            current_dim = hidden_dim

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            The node embeddings of each node in the graph. For N nodes with d-dimensional features, will be of size [N, d].
            E.g. For three nodes n1, n2, and n3 with corresponding feature vectors d1, d2, and d3, x should be [d1, d2, d3].
        edge_index : Tensor
            Node-pairs for which there exists an edge between. For E edges, edge_index will be of dimensions [2, E]
            For example, if a node exists from node n1 to n2, edge index should contain [..., [n1, n2], ...].
        edge_weight : Tensor
            Corresponding weights for the edges in edge_index. For E edges, edge_weights should be of dimention [E].

        Returns
        -------
        Tensor
            Resulting node embeddings from the final GCN layer. Of dimension [N, last_layer_size].
        """
        for conv_layer in self.conv_layers[:-1]:
            x = conv_layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        # Last layer, no activation
        x = self.conv_layers[-1](x, edge_index, edge_weight=edge_weight)
        return x


class GATv2Encoder(nn.Module):
    """
    Class to create encoder architectures comprised of Graph Attention Network v2 layers.
    """
    def __init__(
            self, 
            in_channels, 
            layer_sizes,
            heads=1, 
            dropout=0.0, 
            edge_dim=1, 
            concat=True
        ):
        """
        Create/initialise GATv2Encoder object of custom depth.

        Parameters
        ----------
        in_channels : int
            Initial feature vector dimensions. E.g. if nodes contain 10 features, in_channels should be set to 10.
            Alternatively, in_channels can be set to -1 to infer the number of node features dynamically.
        layer_sizes : List[int]
            Output dimension of each GCN layer, where each integer element in layer_sizes is the output dimension of the i-th layer.
            E.g. layer_sizes = [64,32,16] would create three GCN layers with output dimensions 64, 32, and 16 respectively.
        heads : int
            Number of attention heads to use.
        dropout : float
            Dropout probability of attention coefficients.
        edge_dim : int
            Edge feature dimensions.
        concat : bool
            Whether to concatenate attention heads (if True) or average them (if False).

        Returns
        -------
        None        
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        current_dim = in_channels
        for hidden_dim in layer_sizes:
            # Create one GATv2Conv per hidden size
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=current_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
            current_dim = hidden_dim * heads if (concat==True) else hidden_dim

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            The node embeddings of each node in the graph. For N nodes with d-dimensional features, will be of size [N, d].
            E.g. For three nodes n1, n2, and n3 with corresponding feature vectors d1, d2, and d3, x should be [d1, d2, d3].
        edge_index : Tensor
            Node-pairs for which there exists an edge between. For E edges, edge_index will be of dimensions [2, E]
            For example, if a node exists from node n1 to n2, edge index should contain [..., [n1, n2], ...].
        edge_weight : Tensor
            Corresponding weights for the edges in edge_index. For E edges, edge_weights should be of dimention [E].

        Returns
        -------
        Tensor
            Resulting node embeddings from the final GCN layer. Of dimension [N, last_layer_size].
        """
        edge_attr = edge_weight.view(-1, 1)
        for conv_layer in self.conv_layers[:-1]:
            x = conv_layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
        # Last layer - no activation function.
        x = self.conv_layers[-1](x, edge_index, edge_attr=edge_attr)
        return x


class GraphSAGEEncoder(nn.Module):
    """
    Class to create encoder architectures comprised of Graph Search and Aggregate layers.
    """
    def __init__(self, in_channels, layer_sizes, aggr="mean"):
        """
        Create/initialise dGCNEncoder object of custom depth.

        Parameters
        ----------
        in_channels : int
            Initial feature vector dimensions. E.g. if nodes contain 10 features, in_channels should be set to 10.
            Alternatively, in_channels can be set to -1 to infer the number of node features dynamically.
        layer_sizes : List[int]
            Output dimension of each GCN layer, where each integer element in layer_sizes is the output dimension of the i-th layer.
            E.g. layer_sizes = [64,32,16] would create three GCN layers with output dimensions 64, 32, and 16 respectively.
        aggr : str
            Method for aggregating neighbouring node features. Can be "mean", "max", or "lstm".

        Returns
        -------
        None        
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()

        current_dim = in_channels
        for out_dim in layer_sizes:
            self.conv_layers.append(
                SAGEConv(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    aggr=aggr
                )
            )
            current_dim = out_dim

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass.

        GraphSAGE's original implementation ignores edge weights, therefore `edge_weights` is not used.

        Parameters
        ----------
        x : Tensor
            The node embeddings of each node in the graph. For N nodes with d-dimensional features, will be of size [N, d].
            E.g. For three nodes n1, n2, and n3 with corresponding feature vectors d1, d2, and d3, x should be [d1, d2, d3].
        edge_index : Tensor
            Node-pairs for which there exists an edge between. For E edges, edge_index will be of dimensions [2, E]
            For example, if a node exists from node n1 to n2, edge index should contain [..., [n1, n2], ...].
        edge_weight : Tensor
            IGNORED. Corresponding weights for the edges in edge_index. PyG's SAGEConv implementation does not use edge weights.
            however parameter is kept for consistency and ease of integration.

        Returns
        -------
        Tensor
            Resulting node embeddings from the final GCN layer. Of dimension [N, last_layer_size].
        """
        for layer in self.conv_layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.conv_layers[-1](x, edge_index)
        return x


class TransformerEncoder(nn.Module):
    """
    Class to create encoder architectures comprised of Unified Message Passing layers.
    """
    def __init__(
            self, 
            in_channels,
            layer_sizes,
            heads=1,
            dropout=0.0,
            edge_dim=1,
            concat=True
        ):
        """
        Create/initialise GATv2Encoder object of custom depth.

        Parameters
        ----------
        in_channels : int
            Initial feature vector dimensions. E.g. if nodes contain 10 features, in_channels should be set to 10.
            Alternatively, in_channels can be set to -1 to infer the number of node features dynamically.
        layer_sizes : List[int]
            Output dimension of each GCN layer, where each integer element in layer_sizes is the output dimension of the i-th layer.
            E.g. layer_sizes = [64,32,16] would create three GCN layers with output dimensions 64, 32, and 16 respectively.
        heads : int
            Number of attention heads to use.
        dropout : float
            Dropout probability of attention coefficients.
        edge_dim : int
            Edge feature dimensions.
        concat : bool
            Whether to concatenate attention heads (if True) or average them (if False).

        Returns
        -------
        None        
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        current_dim = in_channels
        for out_dim in layer_sizes:
            self.conv_layers.append(
                TransformerConv(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=concat
                )
            )
            # If concat=True, the output dimension of each layer is out_dim * heads
            # If concat=False, the output dimension is out_dim
            current_dim = out_dim * heads if concat else out_dim

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            The node embeddings of each node in the graph. For N nodes with d-dimensional features, will be of size [N, d].
            E.g. For three nodes n1, n2, and n3 with corresponding feature vectors d1, d2, and d3, x should be [d1, d2, d3].
        edge_index : Tensor
            Node-pairs for which there exists an edge between. For E edges, edge_index will be of dimensions [2, E]
            For example, if a node exists from node n1 to n2, edge index should contain [..., [n1, n2], ...].
        edge_weight : Tensor
            Corresponding weights for the edges in edge_index. For E edges, edge_weights should be of dimention [E].

        Returns
        -------
        Tensor
            Resulting node embeddings from the final UniMP layer. Of dimension [N, last_layer_size].
        """
        edge_attr = edge_weight.view(-1, 1)
        
        for conv_layer in self.conv_layers[:-1]:
            x = conv_layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
        # Last layer
        x = self.conv_layers[-1](x, edge_index, edge_attr=edge_attr)
        return x
    

class SimpleEncoder(nn.Module):
    """
    Pseudo encoder that passes original node features without modification.
    
    Used to compare GNN appproaches against no GNNs.
    """
    def forward(self, x, edge_index, edge_weights):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            The node features of each node in the graph. 
        edge_index : Optional(Tensor)
            IGNORED. Node-pairs for which there exists an edge between.
        edge_weight : Optional(Tensor)
            IGNORED. Corresponding weights for the edges in edge_index.

        Returns
        -------
        Tensor
            Original node features.
        """
        return x
