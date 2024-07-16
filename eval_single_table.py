import os.path

from grid_index.multidimensional_grid import *
import data_location_variables
from Table import Table, convert_row
import pandas as pd
import datetime
import pickle
import numpy as np
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

tpch_headers = {
    "customer": ["C_CUSTKEY", "C_NAME", "C_ADDRESS", "C_NATIONKEY", "C_PHONE", "C_ACCTBAL", "C_MKTSEGMENT"]
}


def read_tables_new(table_names, dataset_name = "tpch", header_flag = False, separator = "|"):

    cols_for_removal = dict()
    cols_for_removal["customer"] = {7}  # only do not read the c_comment column, if present
    table_names = set(table_names)
    header = list()
    tables = {}
    for table_name in table_names:
        if table_name == 'customer':
            """ custom header for customer """
            header = ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment"]
        else:
            header = []

        if header_flag == False:
            rows_pandas = pd.read_csv(data_location_variables.data_location + dataset_name + "/" + table_name + ".csv",
                                       delimiter=separator, header=None, escapechar="\\")
        else:
            rows_pandas = pd.read_csv(data_location_variables.data_location + dataset_name + "/" + table_name + ".csv",
                                       delimiter=separator, escapechar="\\")

        if 'squawk' in rows_pandas.columns:
            rows_pandas['squawk'] = rows_pandas['squawk'].astype(object)

        if rows_pandas.size != 0:
            rows = []

            cols_removal = []
            if table_name in cols_for_removal:
                cols_removal = cols_for_removal[table_name]

            column_type = []
            actual_col_indx = 0
            for tmp_col_id, type in enumerate(rows_pandas.dtypes):
                if tmp_col_id in cols_removal:
                    continue

                if type == "int64" or type == "int32" or type == "float64" or type == "float32":
                    if rows_pandas.columns[tmp_col_id] == 'account':
                        """this is for paymentsless, when we have account it needs to be treated as a string not int"""

                        column_type.append("string")

                        rows_pandas[rows_pandas.columns[tmp_col_id]] = rows_pandas[
                            rows_pandas.columns[tmp_col_id]].astype('object').fillna("")

                    else:
                        column_type.append("float")

                        rows_pandas[rows_pandas.columns[tmp_col_id]] = rows_pandas[rows_pandas.columns[tmp_col_id]].\
                            astype('float32').fillna(0.)
                else:
                    # print(header[actual_col_indx])
                    if (rows_pandas.columns[tmp_col_id] in ['regvaliddate', 'regexpirationdate', 'documentdate', 'o_orderdate']) \
                            or (len(header) > 0 and header[actual_col_indx] in ['regvaliddate', 'regexpirationdate', 'documentdate', 'o_orderdate']):
                        # print('processing date')
                        """handle dates"""
                        column_type.append("date")

                        rows_pandas[rows_pandas.columns[tmp_col_id]] = pd.to_datetime(rows_pandas[
                            rows_pandas.columns[tmp_col_id]].fillna('10/01/2023'), infer_datetime_format=True).dt.strftime("%Y-%m-%d")
                    else:
                        column_type.append("string")
                        if table_name == 'flight' and tmp_col_id == 5:
                            """
                                this means that we are working with the column squawk and needs to be treated as float
                            """
                            rows_pandas[rows_pandas.columns[tmp_col_id]] = rows_pandas[
                                rows_pandas.columns[tmp_col_id]].astype('object').fillna(0.)#.replace("", 0., regex=True)
                        else:
                            rows_pandas[rows_pandas.columns[tmp_col_id]] = rows_pandas[
                                rows_pandas.columns[tmp_col_id]].astype('object').fillna("")
                print(f'{type} {column_type}')
                actual_col_indx += 1


            for drop_col_indx in sorted(list(cols_removal), reverse=True):
                print(f'dropping column {drop_col_indx}')
                rows_pandas = rows_pandas.drop(rows_pandas.columns[drop_col_indx], axis=1)

            rows = rows_pandas.values.tolist()

            if len(header) == 0:
                if not header_flag:
                    header = [str(i) for i in range(len(rows[0]))]

                    print("The header for table " + table_name + " is " + str(header))
                else:
                    header = rows_pandas.columns

            # added for tables to ignore
            if len(cols_for_removal) > 0:
                new_header = []
                for c_j, cell in enumerate(header):
                    if c_j not in cols_removal:
                        new_header.append(cell)
                header = new_header


            print(rows_pandas.dtypes)
            print(column_type)
            print(header)

            table = Table(table_name, header, rows, column_type)
            tables[table_name] = table

            print("=====")
            print(header)

    print('Okay created all tables')
    return tables

def convert_tables(tables):
    for table_name in tables.keys():

        table_i = tables[table_name]
        table_i.index_enabled = True
        # If the number of values is large then do a hash
        column_mapper = dict()
        min_max = dict()
        set_not_in = set()
        for i in range(len(table_i.header)):
            print("Processing header " + str(table_i.header[i]))
            col_i_data = table_i.rows[:, i]

            # If the data is numerical we do not need anything
            if table_i.header[i] not in set_not_in and (isinstance(col_i_data[0], int) or isinstance(col_i_data[0], float)):
                print("This is a numerical value we will continue")
                min_max[table_i.header[i]] = [min(col_i_data), max(col_i_data)]
                continue

            data = set(col_i_data)
            # If the number of values is really small store a mapper but we also sort it
            if len(data) < 20000000:
                if table_i.header[i] in ['regexpirationdate', 'regvaliddate', 'documentdate']:
                    """handling of date columns"""
                    data = sorted(data, key=lambda date: datetime.datetime.strptime(date, "%Y-%m-%d"))
                else:
                    data = sorted(data)

                column_mapper_i = dict()
                path_file_name = './col_mappings_str_date/' + table_name + '/' + table_i.header[i] + '.csv'
                do_this_saving = True
                if do_this_saving and os.path.exists(path_file_name):
                    print('column mapping exists')
                    # index_mapping = dict()
                    with open(path_file_name, 'r') as read_file:
                        for line_indx, line in enumerate(read_file):
                            if line_indx > 0:
                                line_split = line.split('|')
                                # original = line_split[0].strip()
                                original = line_split[0]

                                mapped = int(line_split[1].strip())
                                # mapped = int(line_split[1])
                                column_mapper_i[original] = mapped
                else:
                    for d_i, d in enumerate(data):
                        column_mapper_i[d] = int(d_i)

                    if do_this_saving:
                        with open(path_file_name, 'w+') as mapping_file:
                            mapping_file.write('original|mapped\n')
                            # mapping_file.flush()
                            # num_files_writing = 0

                            for original_val, mapped_val in column_mapper_i.items():
                                mapping_file.write(original_val + '|' + str(mapped_val) + '\n')

                            mapping_file.flush()

                column_mapper[table_i.header[i]] = column_mapper_i
                min_max[table_i.header[i]] = [min(column_mapper_i.values()), max(column_mapper_i.values())]

                continue

        table_i.column_mapper = column_mapper
        # the same info is stored in the two variants
        table_i.min_max = min_max

        range_start = []
        range_end = []
        min_max = table_i.min_max
        for col_name in table_i.header:
            col_min, col_max = min_max[col_name]
            range_start.append(col_min)
            range_end.append(col_max)
        table_i.range_start = np.array(range_start)
        table_i.range_end = np.array(range_end)
        # table_i.range_start = range_start
        # table_i.range_end = range_end

    for table_name in tables.keys():
        table_i = tables[table_name]
        new_rows = []
        for i, row in enumerate(table_i.rows):
            if (i % 10000) == 0:
                print(i)
            new_row = convert_row(row, table_i)

            new_rows.append(np.array(new_row))

        table_i.new_rows = new_rows


if __name__ == '__main__':
    """which tables need to be read"""
    t = 'customer'
    # some setting for grid index
    dimensions_to_ignore = []
    ranges_per_dimension = []
    column_indxs_for_ar_model_training = []
    column_names = ''
    header_flag = False
    if t == 'customer':
        # "customer": ["C_CUSTKEY", "C_NAME", "C_ADDRESS", "C_NATIONKEY", "C_PHONE", "C_ACCTBAL", "C_MKTSEGMENT"]
        dimensions_to_ignore = [1, 2, 4, 6]
        # this is both the dimensions to ignore and the dimensions to delete which are set in the read_tables method
        dims_to_delete_for_querying = [1, 2, 4, 6]
        """for the ar model, we are only interested in training over the columns that can have equalities and not others"""
        ranges_per_dimension = [5, 0, 0, 5, 0, 5, 0]
        column_indxs_for_ar_model_training = [1, 2, 4, 6]
        column_names = 'grid_cell,c_name,c_address,c_phone,c_mktsegment'
        separator = '|'



    if data_location_variables.read_parsed_dataset and os.path.exists('./table_objects/{}.pickle'.format(t)):
        with open('./table_objects/{}.pickle'.format(t), 'rb') as existing_obj:
            table_of_interest = pickle.load(existing_obj)
    else:
        # read the data from the table
        tables = read_tables_new([t], dataset_name=data_location_variables.dataset_name, header_flag=header_flag, separator=separator)

        # convert the table
        convert_tables(tables)

        table_of_interest = tables[t]
        # store the dataset
        if data_location_variables.store_parsed_dataset:
            with open('./table_objects/{}.pickle'.format(t), 'wb') as store_obj:
                pickle.dump(table_of_interest, store_obj, protocol=pickle.HIGHEST_PROTOCOL)


    relevant_ranges_only = [range_tmp
                            for range_indx, range_tmp in enumerate(ranges_per_dimension)
                            if range_indx not in column_indxs_for_ar_model_training]


    if not data_location_variables.load_existing_grid:
        print("Creating a new index")
        # get the rows in the right format
        example_data = list(zip(*table_of_interest.new_rows))

        grid_index = MultidimensionalGrid(example_data, relevant_ranges_only, cdf_based=True,
                                          dimensions_to_ignore=dimensions_to_ignore,
                                          column_indxs_for_ar_model_training=column_indxs_for_ar_model_training,
                                          column_names=column_names,
                                          table_name=table_of_interest.table_name,
                                          table_size=len(table_of_interest.new_rows))

        if data_location_variables.store_grid:
            # grid_index.estimator = None # TODO: uncomment to store just the grid without the ar model to measure grid memory
            with open('index_' + str(table_of_interest.table_name) + data_location_variables.grid_ar_name,
                      'wb') as handle:
                pickle.dump(grid_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading an existing index: ")
        with open('index_' + str(t) + data_location_variables.grid_ar_name, 'rb') as handle:
            grid_index = pickle.load(handle)

    # queries_file_name = "queries_generator/queries/{}_tpch-customer-50-queries.pickle" # ~ 50 queries
    queries_file_name = "queries_generator/queries/{}_tpch-customer-300-queries.pickle" # 300 queries
    all_queries_min = list()
    with open(queries_file_name.format('min'), 'rb') as read_file:
        all_queries_min = pickle.load(read_file)

    all_queries_max = list()
    with open(queries_file_name.format('max'), 'rb') as read_file:
        all_queries_max = pickle.load(read_file)

    query_results = list()
    # with open('queries_generator/queries/tpch-customer-50-queries.sql', 'r') as queries_path: # ~ 50 queries
    with open('queries_generator/queries/tpch-customer-300-queries.sql', 'r') as queries_path:
        for line_id, line in enumerate(queries_path):
            query, query_result = line.split("|")
            query_result = int(query_result)

            query_results.append(query_result)

    # take the column names
    column_names_split = [c_name for c_name in column_names.split(',')]

    grid_index_query_min = list()
    grid_index_query_max = list()
    ar_model_queries = list()

    ar_model_columns_indexes = list()
    for ar_model_cols in column_names_split[1:]:
        if ar_model_cols.upper() in table_of_interest.header:
            ar_model_columns_indexes.append(table_of_interest.header.index(ar_model_cols.upper()))
        elif ar_model_cols.lower() in table_of_interest.header:
            ar_model_columns_indexes.append(table_of_interest.header.index(ar_model_cols.lower()))

    column_indexes_of_cols_not_in_ar_model = list()
    for col_name in table_of_interest.header:
        if col_name.lower() not in column_names_split:
            column_indexes_of_cols_not_in_ar_model.append(table_of_interest.header.index(col_name))
    print('indexes of cols that are not in ar model:')
    print(column_indexes_of_cols_not_in_ar_model)

    print('column indexes of ar model:')
    print(ar_model_columns_indexes)
    print()
    print(f'there are {len(all_queries_min)} queries')

    dimensions_to_ignore.sort(reverse=True)
    dims_to_delete_for_querying.sort(reverse=True)

    start_processing_queries = time.time()
    for query_indx, query_min in enumerate(all_queries_min):
        query_max = all_queries_max[query_indx]
        '''part for the ar_model'''
        ar_model_query = [None] * (len(column_names_split) - 1)
        ar_model_query = [-1] + ar_model_query


        exact_query_min = query_min.copy()
        exact_query_max = query_max.copy()

        '''part for the grid index'''
        grid_index_min = query_min.copy()
        grid_index_max = query_max.copy()

        for col_indx, col_name in enumerate(column_names_split[1:]):
            """
                take the query value for the columns that are mapped and 
                map it to the value used internally for the models
            """
            if exact_query_min[ar_model_columns_indexes[col_indx]] is not None:
                # for the min part
                if col_name.upper() in table_of_interest.column_mapper.keys():
                    min_val_map = table_of_interest.column_mapper[col_name.upper()][
                        exact_query_min[ar_model_columns_indexes[col_indx]]]
                elif col_name.lower() in table_of_interest.column_mapper.keys():
                        min_val_map = table_of_interest.column_mapper[col_name.lower()][
                            exact_query_min[ar_model_columns_indexes[col_indx]]]

                exact_query_min[ar_model_columns_indexes[col_indx]] = min_val_map

                # for the max part
                exact_query_max[ar_model_columns_indexes[col_indx]] = min_val_map

                # for the ar model need to put it on another position
                ar_model_query[col_indx+1] = min_val_map

        """
            no need to store the dimensions that we will be ignoring for the creation of the grid 
        """
        for dim_to_delete in dims_to_delete_for_querying:
            del grid_index_min[dim_to_delete]
            del grid_index_max[dim_to_delete]

        for dim, _ in enumerate(grid_index_min):
            if grid_index_min[dim] is None:
                grid_index_min[dim] = grid_index.min_per_dimension[dim]
                grid_index_max[dim] = grid_index.max_per_dimension[dim]
            else:
                col_name = table_of_interest.header[column_indexes_of_cols_not_in_ar_model[dim]]

                """need to check dates"""
                if col_name.upper() in table_of_interest.column_mapper.keys():
                    min_val_map = table_of_interest.column_mapper[col_name.upper()][
                        grid_index_min[dim]]

                    max_val_map = table_of_interest.column_mapper[col_name.upper()][
                        grid_index_max[dim]]

                    grid_index_min[dim] = min_val_map
                    grid_index_max[dim] = max_val_map
                elif col_name.lower() in table_of_interest.column_mapper.keys():
                    min_val_map = table_of_interest.column_mapper[col_name.lower()][
                        grid_index_min[dim]]

                    max_val_map = table_of_interest.column_mapper[col_name.lower()][
                        grid_index_max[dim]]

                    grid_index_min[dim] = min_val_map
                    grid_index_max[dim] = max_val_map

        grid_index_query_min.append(grid_index_min)
        grid_index_query_max.append(grid_index_max)
        ar_model_queries.append(ar_model_query)
    end_processing_queries = time.time() - start_processing_queries
    print(f'Time for processing queries in right format {end_processing_queries * 1000 } MS')
    # print('The max and min queries are: ')
    # print(grid_index_query_min)
    # print(grid_index_query_max)
    # exit(1)

    print('----ESTIMATING----')
    q_error = list()
    time_total = list()
    time_total_exact = list()
    for query_indx, query_min in enumerate(grid_index_query_min):
        query_max = grid_index_query_max[query_indx]
        ar_query = ar_model_queries[query_indx]

        query_estimation, estimation_time = grid_index.single_table_estimation(query_min_vals=query_min,
                                                                                          query_max_vals=query_max,
                                                                                          values_ar_model=ar_query,
                                                                                          columns_for_ar_model=column_names_split)

        print(f'Query estimated result is {query_estimation} for {estimation_time} MS')
        time_total.append(estimation_time)

        # exact query result
        true_result = query_results[query_indx]
        print(f'Query true result: {true_result}')
        if torch.is_tensor(query_estimation):
            query_estimation = query_estimation.item()

        q_error.append(max(query_estimation / true_result, true_result / query_estimation))


    print(f'Average q-error {np.average(q_error)}')
    print(f'Median q-error {np.quantile(q_error, .5)}')
    print(f'90 percentile q-error {np.quantile(q_error, .9)}')
    print(f'99 percentile q-error {np.quantile(q_error, .99)}')
    print(f'Max q-error {np.quantile(q_error, 1.)}')
    print(f'Our approach avg execution time: {np.average(time_total)} MS')
    print(f'Our approach median execution time: {np.quantile(time_total, .5)} MS')
    print(f'Our approach minimal execution time: {np.quantile(time_total, .0)} MS')
    print(f'Our approach maximal execution time: {np.quantile(time_total, 1.)} MS')

