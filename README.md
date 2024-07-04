# Grid-AR

### Data

The TPC-H `Customer` table is located in: `datasets/customer/`

## Quick start

Set up a conda environment with dependencies installed:

```bash
# Install Python environment
conda env create -f environment.yml
conda activate grid-ar
# Run commands below inside this directory.
```
The torch version needs to be changed depending on the cuda version.

## Project Setup for Grid-AR 
First, create a `.py` file from `data_location_variables.template` and `PathsVariables.template`. Then put the location to the folder storing the table csv file in `data_location_variables.py` in the variable `data_location`.

Create the following folders:

- `column_mappings_grid` - stores the mapping of the grid cells to consecutive integers starting from `0`
- `compressor_elems` - stores the compressor element that performs the compression of columns with many unique values
- `models_vs_ar_modelps` - stores the AR model 

For large datasets, in `data_location_variables.py`, change between storing and creating Grid-AR. 
To store the pre-processed dataset and store Grid-AR, set `store_parsed_dataset` and `store_grid` to `True`.
To load a pre-processed dataset and an existing Grid-AR estimator, set `read_parsed_dataset` and `load_existing_grid` to `True`.
When creating a new Grid-AR estimator, set the name for the estimator by changing `grid_ar_name = ''`. 


Everything is set up for training a new CDF-based Grid-AR. 
To load an existing model, change `load_existing_grid=True` and `grid_ar_name` in `data_location_variables.py` and the number of ranges per column.
Also, the Grid-AR structure with the AR model and all required info will be stored for consecutive invocations.
For different buckets per column, a different Grid-AR structure with AR model needs to be created. 
The number of buckets per column can be set in `eval_single_table.py` under the variable `ranges_per_dimension`.


To change the number of **epochs** for which the model is trained you can do that in `train_model.py` in the parser argument `--epochs`, default is `10`.
By default, the model will use compression.

## Experiments
To measure the memory consumption of the grid structure, remove the estimator before saving. 
The memory consumption of the AR model is part of the name of the saved model.

To repeat the experiments for single table queries and get the results for Figure 4, Table 2, Table 3, and Table 4, run `eval_single_table.py` as is.

To repeat the experiments for range join queries (Table 6-7, Figure 6), run `eval_multi_range_joins.py` as is.
- To evaluate the different range join types and number of tables included in the range join, uncomment the respective query file names.

