## Data Collection and Pre-processing

### Contents
 - [Usage](#usage)
 - [Definitions](#definitions)
   - [csv_foodwebs/](#csv_foodwebs)
   - [csv_node_features/](#csv_node_features)
   - [final_foodwebs/](#final_foodwebs)
   - [final_node_features/](#final_node_features)
   - [json_node_features/](#json_node_features)
   - [ohe_node_features/](#ohe_node_features)
   - [import_node_features.R](#import_node_features.R)
   - [scaler.py](#scaler.py)
   - [unify_taxized_datasets.py](#unify_taxized_datasets.py)

<h3 id=usage>
    Usage
</h3>

Whilst scripts can be run individually to collect node features and pre-process food web data, a supplementary script `data_processing.sh` has been provided in the root folder of this repository. `data_processing.sh` calls the data collection and pre-processing scripts in turn, providing a single command to perform the entirety of this pipeline. To use this, please run the following command in your terminal from the root folder:

```bash
bash data_processing.sh
```

<h3 id=definitions>
    Definitions
</h3>

The below section defines what each folder and script within this directory is used for.

<h4 id=csv_foodwebs>
    Csv_foodwebs/
</h4>

This folder contains all food web files available from [The Web of Life](https://www.web-of-life.es/map.php?type=7). These food webs are pre-processed by [scaler.py](#scaler.py) and transformed into the food webs provided in [final_foodwebs](#final_foodwebs).

<h4 id=csv_node_features>
    Csv_node_features/
</h4>

This folder contains all "species" files for food webs available on [The Web of Life](https://www.web-of-life.es/map.php?type=7). Because these files contain only two useful feature columns (Kingdom, and Role), additional node features are aggregated from third party taxonomic databases prior to the optimisation/training of models. The files contained within [csv_node_features][csv_node_features] serve as input to [import_node_features.R](#import_node_features.R).

<h4 id=final_foodwebs>
    Final_foodwebs/
</h4>
This folder contains the final, pre-processed food webs for use in model optimisation, training, and evaluation. However, it should be noted that, in this research, binary networks are not used and hence while this folder contains FW_005, FW_007, and webs FW_011 to FW_015_04, these are not used.

To see the final data sets as used in hyperparameter optimiastion and training/evaluation, please refer to [final_datasets](../final_datasets/).

<h4 id=final_node_features>
    Final_node_features/
</h4>

This folder contains the final node feature sets to be used in model optimisation training and evaluation, as outputted by [unify_taxized_datasets.py](#unify_taxized_datasets.py).

It should be noted however, that binary networks are out of scope for this research and hence while this folder contains node features for FW_005, FW_007, and webs FW_011 to FW_015_04, these are not used. To see the final data sets as used in hyperparameter optimiastion and training/evaluation, please refer to [final_datasets](../final_datasets/).

<h4 id=json_node_features>
    Json_node_features/
</h4>

This folder contains the outputs of [import_node_features.R](#import_node_features.R) when applied to [csv_node_features/](#csv_node_features) as used within the dissertation, and is the the data behind Table 2. of the paper.

The files within this folder are fed into [unify_taxized_datasets.py](#unify_taxized_datasets.py) to create the files in [ohe_node_features/](#ohe_node_features) and [final_node_features/](#final_node_features).

<h4 id=ohe_node_features>
    Ohe_node_features/
</h4>

This folder contains muti-label binary features produced as an intermediate step in [unify_taxized_datasets.py](#unify_taxized_datasets.py) because the node feature sets contained within this folder do not necessarily contain consistent feature dimensions.

The final step in [unify_taxized_datasets.py](#unify_taxized_datasets.py) standardises the feature dimensions in each food web by aggregating all unique feature columns within the files in [ohe_node_features/](ohe_node_features/), adding any feature columns missing from individual food webs and filling all values with 0's.

<h4 id=import_node_features.R>
    Import_node_features.R
</h4>

This script is used to aggregate species information from the [Global Biodiversity Information Facility (GBIF)](https://www.gbif.org/), [Integrated Taxonomic Information System (ITIS)](https://www.itis.gov/), [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/taxonomy), [World Register of Marine Species (WoRMS)](https://www.marinespecies.org/), and [Encyclopedia of Life (EoL)](https://www.eol.org/) databases and standardise the feature names, before outputting node feature sets to a new directory as JSON files.

You can then run `import_node_features.R` by running the following command from inside `data_processing/`:

```bash
Rscript -e 'renv::run("import_node_features.R")'
```

By default, this script reads the contents of `csv_node_features` and outputs the aggregated node feature sets to a new folder called `json_node_features/`.

<h4 id=unify_taxized_datasets.py>
    Unify_taxized_datasets.py
</h4>

Because the "species" files describing species metadata available on [The Web of Life](https://www.web-of-life.es/map.php?type=7) can contain multiple rows for a single species, [unify_taxized_datasets.py](unify_taxized_datasets.py) identifies where this occurs and converts categorical features to multi-label binary features as models are unable to accept string features as input. The resulting node feature sets are output as JSON files to a new (or specified) folder (by default: [final_node_features/](final_node_features)). To use this script, please run the following from the `data_processing/` folder in your terminal after running `import_node_features.R`:

```bash
python unify_taxized_datasets.py --input_dir {output directory of import_node_features.R} --out_dir {folder path to output final node features to}
```

E.g.

```bash
python unify_taxized_datasets.py --input_dir json_node_features --out_dir final_node_features
```

<h4 id=scaler.py>
    Scaler.py
</h4>

Because interaction strengths are quantified using differing methodologies across the food web datasets, `scaler.py` uses the min-max scaler to bring all interactions in the range of [0,1], thereby preventing food webs with large edge weights from dominating the error metrics. To run this script independently, use the following command from inside `data_processing/`:

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

