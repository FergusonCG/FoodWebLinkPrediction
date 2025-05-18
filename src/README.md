## src

This folder contains the main implementation of link weight prediction models for food webs, as well as command-line interfaces for performing hyperparameter optimisation and the final train/test pipeline in [hyperparameter_optimisation.py](hyperparameter_optimisation.py) and [train_final_models.py](train_final_models.py) respectively.

### Contents
 - [config.py](#config.py)
 - [data.py](#data.py)
 - [decoders.py](#decoders.py)
 - [encoders.py](#encoders.py)
 - [hyperparameter_optimisation.py](#hyperparameter_optimisation.py)
 - [models.py](#models.py)
 - [train_final_models.py](#train_final_models.py)
 - [utils.py](#utils.py)


<h3 id=config.py>
    config.py
</h3>

A minimal script configuring which device to use if CUDA is enabled (recommended when training GNN-based architectures). If CUDA is not available, models are trained using the device's CPU.

<h3 id=data.py>
    data.py
</h3>

Contains functionality for loading food web data from JSON files, as contained within [final_datasets](../final_datasets/) and creating a training set from loaded food webs.

The functionality within this script is used by [hyperparameter_optimisation.py](hyperparameter_optimisation.py) and [train_final_models.py](train_final_models.py).

<h3 id=decoders.py>
    decoders.py
</h3>

Enables the dynamic creation of decoder architectures which utilise Multi-Layer Perceptrons (MLPs) to perform regression on edge weights, using the node embeddings produced by an encoder from [encoders.py](#encoders.py).

The `LinkWeightDecoder` class defined in this file is used within [models.py](#models.py) to create end-to-end link weight prediction models.

<h3 id=encoders.py>
    encoders.py
</h3>

Enables the dynamic creation of Graph Neural Network (GNN) encoder architectures. The encoders used in this research are Graph Convolutional Networks (GCNs), Graph Sample and Aggregate (GraphSAGE), Graph Attention Networks v2 (GATv2), Unified Message-Passing Graph Transformers (UniMP). In addition to the above, a custom `SimpleEncoder` is implemented, which returns the original node features without performing any node embedding, to simplify the construction of architectures which do not utilise GNNs.

The respective classes for each encoder architecture within this file are used within [models.py](#models.py) to create end-to-end link weight prediction models.

<h3 id=hyperparameter_optimisation.py>
    hyperparameter_optimisation.py
</h3>

This script contains the class `GNNHPO` implementing the hyperparameter optimisation methodology, as well as a command-line interace (CLI) allowing hyperparameter optimisation to be performed via the command line if provided with pre-processed networks and node features (as in [final_datasets/](../final_datasets/)).

This script imports the functionality defined within [config.py](#config.py), [data.py](#data.py), [models.py](#models.py), and [utils.py](#utils.py).

To use this script independently via the CLI, please use the following command:

```bash
python src/hyperparameter_optimisation.py --food_webs_path {folder containing food web edge weight matrix JSON files} --node_features_path {folder containing food web node feature JSON files} --pca_components {number of PCA components to use or 0 for no PCA} --out_dir {folder to output results to}
```

For example:

```bash
python src/hyperparameter_optimisation.py --food_webs_path final_datasets/ high_feature_coverage-food_webs/ --node_features_path final_datasets/high_feature_coverage-node_features/ --pca_components 0 --out_dir results/optimisation_results/high_feature_coverage/0_pca_components
```

<h3 id=models.py>
    models.py
</h3>

This file enables the dynamic creation of end-to-end link weight prediction models, using the encoder and decoder architectures defined in [encoders.py](#encoders.py) and [decoders.py](#decoders.py).

These models are then instantiated within [hyperparameter_optimisation.py](hyperparameter_optimisation.py) and [train_final_models.py](train_final_models.py).

<h3 id=train_final_models.py>
    train_final_models.py
</h3>

This script implements the training methodology and final evaluation of model configurations defined within an external file. The intended use for this is to pass in the best configurations found during hyperparameter optmisation; however, if you wish to use a custom architecture, please create a JSON file following the format:

```json
[
    {
        "n_encoder_layers": int,
        "encoder_layer_width_0": int,
        ...,
        "n_decoder_layers": int,
        "decoder_layer_width_0": int,
        ...,
        "lr": float,
        "encoder_type": str,
        "pca_components": int,
        ...
    }
]
```

Where "encoder_layer_width_{i}" should be defined from `i = 0` to `i = n_encoder_layers - 1`, "encoder_layer_width_{j}" should be defined from `j = 0` to `j = n_decoder_layers - 1`, and the final "..." should contain any `encoder_type` specific kwargs.

`encoder_type` can be either "gcn", "graphsage", "gatv2", "transformer", or "simple". "gcn" encoder_types have no kwargs; "graphsage" encoder_types have one additional kwarg "graphsage_aggregator" which can be any [PyTorch Geometric aggregation method](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.Aggregation.html); "gatv2" has two additional kwargs, "gatv2_heads" - which is an integer detailing how many attention heads to use, and "gatv2_dropout" - which is a float detailing the attention drop-out rate; "transformer" has two additional kwargs, "transformer_heads" - which is an integer detailing how many attention heads to use, and "gatv2_dropout" - which is a float detailing the attention drop-out rate; and "simple" encoders can omit "n_encoder_layers" and any "encoder_layer_width_{k}" keys as they do not perform node embedding. Please refer to one of the `summary.json` files in [results/](../results/)'s sub-folders for examples of this.

To run the script from a command-line, please use the following command:

```bash
python src/train_final_models.py --webs_path {path to folder containing pre-processed food webs} --species_path {path to folder containing pre-processed node features} --best_params {path to model configuration(s) file} --out_dir {path to folder to save results to}
```

For example:

```bash
python src/train_final_models.py --webs_path final_datasets/all_weighted-food_webs/ --species_path final_datasets/all_weighted-node_features/ --best_params results/optimisation_results/all_weighted/0_pca_components/hpo_best_configs.json --out_dir results/all_weighted_networks/0_pca_components-subgraph
```

<h3 id=utils.py>
    utils.py
</h3>

This file contains utility scripts and wrappers used in  [hyperparameter_optimisation.py](#hyperparameter_optimisation.py) and [train_final_models.py](#train_final_models.py), including but not limited to: implementation of the metrics used to evaluate model performance, creation of end-to-end models from JSON configurations (as output by [hyperparameter_optimisation.py](#hyperparameter_optimisation.py)), and the training and evaluation routines.
