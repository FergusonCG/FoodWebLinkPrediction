cd data_processing

# Aggregate node features using Taxize
Rscript import_node_features.R

# Convert node features to consistent format
#   *if your data is stored in different directories, please change arguments accordingly 
python unify_taxized_datasets.py --input_dir json_node_features --out_dir final_node_features

# Scale food web graphs and convert to JSON
python scaler.py csv_foodwebs --out_dir final_foodwebs
