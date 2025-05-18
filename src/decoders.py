import torch
import torch.nn as nn


class LinkWeightDecoder(nn.Module):
    """
    Class to create decoder architectures dynamically.

    A decoder that can have an arbitrary MLP with user-defined
    layer sizes. For example, layer_sizes=[64, 64] means two hidden
    layers of size 64 each. Or [32, 64, 64, 32], etc.
    """
    def __init__(self, in_channels, layer_sizes):
        """
        Instantiate class.

        By taking a list of hidden layer sizes (`layer_sizes`) as input, multi-layered decoder architectures
        can be created, wherein each element `i` in `layer_sizes` denotes how many neurons to use for the `i`-th
        layer. 

        ReLU activation functions are used between each fully connected neural layer to provide non-linearity.

        A final, single-neuron layer is appended at the end to perform regression.

        Parameters
        ----------
        in_channels : int
            Number of input features to the decoder. As decoders take node-pair embeddings produced from a GNN encoder,
            individual node embeddings are of size `d`, and node-pair embeddings are therefore of size `2d`. The in_channels
            for the decoder must therefore align with the dimensions of node-pair embeddings produced by an encoder.
        layer_sizes : List[int]
            The number of neurons to use in each hidden layer of the decoder. For example, to define a MLP with 3 hidden layers,
            where each layer has 16, 32, and 64 neurons respectively, layer_sizes should be defined as [16, 32, 64].
        """
        super().__init__()
        
        layers = []
        input_dim = 2 * in_channels
        
        # Build hidden layers
        for hidden_dim in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Finally, output layer that predicts a single scalar
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_embeddings, edge_index):
        """
        Forward pass of the MLP.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Matrix of embeddings produced by the encoder for each node in the network. Of size [N, d] where N is the number
            of nodes in the network, and d is the final node embedding dimensions produced by an encoder model.
        edge_index : torch.Tensor
            A [2, E]-dimensional tensor describing the node-pairs for which edges exist between. For example, if an edge exists
            between node IDs 1->2 only in a network, `edge_index` should be [[1], [2]].

        Returns
        -------
        torch.Tensor
            The predicted edge weights, represented as a [E, 1] -dimensional tensor, with each edge being assigned a scalar value
        """
        src = node_embeddings[edge_index[0]]
        dst = node_embeddings[edge_index[1]] 
        node_pair_embedding = torch.cat([src, dst], dim=1)
        return self.mlp(node_pair_embedding)
