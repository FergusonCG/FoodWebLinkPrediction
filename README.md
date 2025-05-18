## FoodWebLinkPrediction

This repository contains the code behind my Masters dissertation in Artificial Intelligence with the University of Liverpool.

Code has been produced using Python (develoeped using 3.12.3) and R (developed using version 4.3.3) on Ubuntu 22.04 and 24.04 systems.

The code can be split into four sections:
 - A data collection and pre-processing pipeline
 - Exploration of model configurations via hyperparameter optimisation
 - Training and evaluation of final models
 - Model architecture objects

The first three sections can be run from a CLI, whilst the model architecture objects implement configurable model architectures and the required functionality to build, train, and evaluate configured architectures.

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

Food webs and the corresponding "species" files have been downloaded from the [Web of Life](https://www.web-of-life.es/map.php?type=7) and are stored within the `data_processing/` folder. Though the species files contain some node features, they only contain three truly applicable features. Additional features are aggregated from external taxonomic databases via lookup mechanisms using the [Taxize R package](https://cran.r-project.org/web/packages/taxize/index.html).

This section is comprised of four main files:
 - [import_node_features.R](data_processing/import_node_features.R)
 - [scaler.py](data_processing/scaler.py)
 - [unify_mapping.py](data_processing/unify_mapping.py)
 - [unify_taxized_datasets](data_processing/unify_taxized_datasets.py)

#### Usage

Whilst scripts can be run individually to collect node features and pre-process food web data, a supplementary script `data_processing.sh` has been provided in the root folder of this repository. `data_processing.sh` calls the data collection and pre-processing scripts in turn, providing a single command to perform the entirety of this pipeline. To use this, please run the following command in your terminal from the root folder:

```bash
bash data_processing.sh
```

#### import_node_features.R

This script is used to aggregate species information from the [Global Biodiversity Information Facility (GBIF)](https://www.gbif.org/), [Integrated Taxonomic Information System (ITIS)](https://www.itis.gov/), [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/taxonomy), [World Register of Marine Species (WoRMS)](https://www.marinespecies.org/), and [Encyclopedia of Life (EoL)](https://www.eol.org/) databases and standardise the feature names, before outputting node feature sets to a new directory as JSON files.

You can then run `import_node_features.R` by running the following command from inside `data_processing/`:

```bash
Rscript -e 'renv::run("import_node_features.R")'
```

By default, this script outputs the aggregated node feature sets to a new folder called `json_node_features/`.

#### scaler.py

Because interaction strengths are quantified using differing methodologies across the food web datasets, `scaler.py` uses the min-max scaler to bring all interactions in the range of [0,1], thereby preventing food webs with large edge weights from dominating the error metrics. To run this script independently, use the following command from `data_processing`:

```bash
python scaler.py {folder containing Web of Life food web files}
```

Which will output the new food webs to a new folder called `final_foodwebs` by default. To specify a different directory to output the final food webs to, please use the `--out_dir` argument:

```bash
python scaler.py {folder containing Web of Life food web files} --out_dir {path to output folder}
```

E.g. 

```bash
python scaler.py csv_foodwebs --out_dir final_foodwebs
```

#### unify_mapping.py

Different taxonomic databases are not always consistent with how they name feature values in their database; for example, null values for a species' taxonomic rank were observed to be either `no_rank`, `no rank` or `norank` across databases. To mitigate this, a mapping table has been provided inside `unify_mapping.py` to standardise these values.

This script is not intended to be used independently, rather `unify_taxized_datasets.py` imports the mapping table and processes synonyms to a standardised value.

#### unify_taxized_datasets.py

This script is used to standardise feature names and values across taxonomic databases based on the mapping table in `unify_mapping.py`. The resulting node feature sets are output as JSON files to a new (or specified) folder. To use this script, please run the following from the `data_processing/` folder in your terminal after running `import_node_features.R`:

```bash
python unify_taxized_datasets.py {output directory of import_node_features.R} {folder path to output final node features to}
```

E.g.

```bash
python unify_taxized_datasets.py json_node_features final_node_features
```

### Hyperparameter optimisation

To find optimal model configurations prior to the final training and evaluation, hyperparameter optimisation is performed individually for each embedding method: Graph Convolutional Networks; GraphSAGE; Graph Attention Network v2s; Unified Message Passing Graph Transformers; and no graph embedder (MLP decoder only). Hyperparameter optimisation logs are output alongside the best configurations found for each model architecture listed above.

Due to the time required to train complex models such as deep UniMP models, hyperparameter optimisation implements early-stopping mechanisms to prune poorly performing trials.

To run hyperparameter optimisation across all architectures, the `run_hpo.py` script has been provided in the `src/` folder, and can be used by running the following command in terminal from the root folder of this repository:

```bash
python src/run_hpo.py --webs_path {path to final JSON food webs folder} --species_path {path to final JSON node features folder}
```

E.g.

```bash
python src/run_hpo.py --webs_path data_processing/final_foodwebs --species_path data_processing/final_node_features
```

### Final training

The final training and evaluation of models should be performed after hyperparameter optimisation, using the best identified configurations. For ease of use, the best identified model configurations during development have been included within the `optimisation_results/` folder.

Once a set of configurations has been acquired, training and evaluation can be run via the following command from the root folder:

```bash
python src/train_final_models



### Troubleshooting
