## FoodWebLinkPrediction

This repository contains the code behind my Masters dissertation in Artificial Intelligence with the University of Liverpool. To view the code on GitHub, please use the following link: https://github.com/FergusonCG/FoodWebLinkPrediction/tree/main.

Code has been produced using Python (developed using 3.12.3) and R (developed using version 4.3.3) on Ubuntu 20.04 and 24.04 systems.

The code can be split into four sections:
 - A data collection and pre-processing pipeline
 - Exploration of model configurations via hyperparameter optimisation
 - Training and evaluation of final models


#### Environment setup

For R code, a `renv` has been added to the `data_processing/` directory containing the necessary packages used in this repository.

For Python code, a `requirements.txt` file has been prepared in the root directory; however, you will need to install PyTorch manually prior to installing the packages in this file as the version you will want depends on your version of CUDA (if applicable). Development used version 2.6 of PyTorch with CUDA 12.6.

**Python environment setup**

It is recommended to use a Python environment manager to prevent package version conflicts. This can be done using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [venv](https://docs.python.org/3/library/venv.html) or any other alternative.

Once you have a new environment set up, please install [PyTorch](https://pytorch.org/get-started/locally/) according to your system specification.

After PyTorch has been installed, you can install the remainin packages using the following command from the root folder of this repository:

```Bash
pip install -r requirements.txt
```

### Data collection and pre-processing

Whilst scripts can be run individually to collect node features and pre-process food web data, a supplementary script `main_data_processing.sh` has been provided in the root folder of this repository. `main_data_processing.sh` calls the data collection and pre-processing scripts in turn, providing a single command to perform the entirety of this pipeline. To use this, please run the following command in your terminal from the root folder:

```bash
bash main_data_processing.sh
```

### Hyperparameter optimisation

To find optimal model configurations prior to the final training and evaluation, hyperparameter optimisation is performed individually for each embedding method: Graph Convolutional Networks; GraphSAGE; Graph Attention Network v2s; Unified Message Passing Graph Transformers; and no graph embedder (MLP decoder only). Hyperparameter optimisation logs are output alongside the best configurations found for each model architecture listed above.

Due to the time required to train complex models such as deep UniMP models, hyperparameter optimisation implements early-stopping mechanisms to prune poorly performing trials.

To run hyperparameter optimisation across all architectures, the `main_hpo.sh` script has been provided, and can be used by running the following command in terminal from the root folder of this repository:

```bash
bash main_hpo.sh
```

### Final training

The final training and evaluation of models should be performed after hyperparameter optimisation, using the best identified configurations. For ease of use, the best identified model configurations during development have been included within the `optimisation_results/` folder.

Once a set of configurations has been acquired, training and evaluation can be run via the following command from the root folder:

```bash
bash main_train_test.sh
```