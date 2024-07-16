import math

import data_location_variables
from eval_single_table import convert_tables, read_tables_new
from grid_index.multidimensional_grid import *
from grid_index import range_filter

import copy
import pickle
import glob

import os
import torch
import parse_multijoin_range_queries as pmj
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    """which tables need to be read"""
    table_folder = 'tpch'
    table_names = ['customer']

    # some setting for grid index
    dimensions_to_ignore = dict()
    join_col_ids = dict()
    ranges_per_dimension = dict()
    column_indxs_for_ar_model_training = dict()
    column_names = dict()
    dims_to_delete_for_querying = dict()
    header_flag = False

    if table_folder == 'tpch':
        # "customer": ["C_CUSTKEY", "C_NAME", "C_ADDRESS", "C_NATIONKEY", "C_PHONE", "C_ACCTBAL", "C_MKTSEGMENT"],
        dimensions_to_ignore['customer'] = [1, 2, 4, 6]

        # this is both the dimensions to ignore and the dimensions to delete which are set in the read_tables method

        dims_to_delete_for_querying['customer'] = [1, 2, 4, 6]
        """for the ar model, we are only interested in training over the columns that can have equalities and not others"""

        # customer
        ranges_per_dimension['customer'] = [5, 0, 0, 5, 0, 5, 0]

        # customer
        column_indxs_for_ar_model_training['customer'] = [1, 2, 4, 6]

        # customer
        column_names['customer'] = 'grid_cell,c_name,c_address,c_phone,c_mktsegment'

        separator = '|'



    if os.path.exists('./table_objects/{}.pickle'.format(table_folder)):
        tables = dict()
        with open('./table_objects/{}.pickle'.format(table_folder), 'rb') as existing_obj:
            tables[table_folder] = pickle.load(existing_obj)
    else:
        tables = read_tables_new(table_names, dataset_name=data_location_variables.dataset_name, header_flag=header_flag, separator=separator)

        # convert the table
        convert_tables(tables)

        if data_location_variables.store_parsed_dataset:
            for t_n in tables.keys():
                with open('./table_objects/{}.pickle'.format(t_n), 'wb') as store_obj:
                    pickle.dump(tables[t_n], store_obj, protocol=pickle.HIGHEST_PROTOCOL)

    for t_n in tables.keys():
        print(t_n)
        print(f'Header {len(tables[t_n].header)}')
        print(f'Number of rows {len(tables[t_n].new_rows)}')

        print('-----------------------------')


    relevant_ranges_only = dict()
    for t_n in tables.keys():
        relevant_ranges_only[t_n] = [range_tmp
                                for range_indx, range_tmp in enumerate(ranges_per_dimension[t_n])
                                if range_indx not in column_indxs_for_ar_model_training[t_n]]
    print('relevant ranges only')
    print(relevant_ranges_only)

    grid_indexes = dict()
    read_data = True
    if not data_location_variables.load_existing_grid:
        for t_n in tables.keys():
            print(f'----------------------WORKING WITH TABLE {t_n}----------------------')
            table_of_interest = tables[t_n]
            example_data = list(zip(*table_of_interest.new_rows))
            # creation of multidimensional grid with ar model
            grid_index = MultidimensionalGrid(example_data, relevant_ranges_only[t_n], cdf_based=True, dimensions_to_ignore=dimensions_to_ignore[t_n],
                                              column_indxs_for_ar_model_training=column_indxs_for_ar_model_training[t_n],
                                              column_names=column_names[t_n], table_name=table_of_interest.table_name,
                                              table_size=len(table_of_interest.new_rows))

            grid_indexes[t_n] = grid_index

            if data_location_variables.store_grid:
                # grid_index.estimator = None  # TO STORE JUST THE MODEL WITHOUT THE ESTIMATOR
                with open('index_' + str(table_of_interest.table_name) + data_location_variables.grid_ar_name + '.pickle', 'wb') as handle:
                    pickle.dump(grid_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        for t_n in tables.keys():
            print("Loading an existing index: ")
            with open('index_' + str(t_n) + data_location_variables.grid_ar_name, 'rb') as handle:
                grid_indexes[t_n] = pickle.load(handle)


    """TPC-H"""
    # tables order when parsing queries
    tables_order = {'customer': 1}
    queries_path = 'queries_generator/range_join_queries/tpch/mj_customer_50_fixed_join_inequality_specific_format.sql'
    # queries_path = 'queries_generator/range_join_queries/tpch/mj_customer_50_join_range_specific_format.sql'
    # queries_path = 'queries_generator/range_join_queries/tpch/mj_customer_50_combination_specific_format.sql'
    # queries_path = 'queries_generator/range_join_queries/tpch/mj_customer_range_size_2_queries_specific_format.sql'
    # queries_path = 'queries_generator/range_join_queries/tpch/mj_customer_inequality_size_2_queries_specific_format.sql'
    """TPC-H"""
    # path to query files
    all_queries = pmj.parse_queries(queries_path, tables_order)
    for query in all_queries:
        query.print()

    for query in all_queries:
        # extract the minimum and maximum for every column of every table for the query
        query_min_vals = dict()
        query_max_vals = dict()
        column_names_split = dict()
        ar_model_columns_indexes = dict()
        column_indexes_of_cols_not_in_ar_model = dict()
        column_names_of_cols_not_in_ar_model = dict()

        for table_name in tables.keys():
            # column names for the table
            column_names_split[table_name] = [c_name for c_name in column_names[table_name].split(',')]

            # sort stuff
            dimensions_to_ignore[table_name].sort(reverse=True)
            dims_to_delete_for_querying[table_name].sort(reverse=True)

            # extract relevant columns for cols in ar model
            ar_model_columns_indexes[table_name] = list()
            for ar_model_cols in column_names_split[table_name][1:]:
                if ar_model_cols.upper() in tables[table_name].header:
                    ar_model_columns_indexes[table_name].append(tables[table_name].header.index(ar_model_cols.upper()))
                elif ar_model_cols.lower() in tables[table_name].header:
                    ar_model_columns_indexes[table_name].append(tables[table_name].header.index(ar_model_cols.lower()))

            # extract relevant columns for cols not in ar model
            column_indexes_of_cols_not_in_ar_model[table_name] = list()
            column_names_of_cols_not_in_ar_model[table_name] = list()
            for col_name in tables[table_name].header:
                if col_name.lower() not in column_names_split[table_name]:
                    column_indexes_of_cols_not_in_ar_model[table_name].append(tables[table_name].header.index(col_name))
                    column_names_of_cols_not_in_ar_model[table_name].append(col_name)

        ar_model_query = dict()
        exact_query_min = dict()
        exact_query_max = dict()
        grid_index_min = dict()
        grid_index_max = dict()
        for t_alias in query.query_predicates_per_table.keys():
            # the original table name
            table_for_alias = query.table_alias_to_table_map[t_alias]
            query_min_vals[t_alias] = [None] * len(tables[table_for_alias].header)
            query_max_vals[t_alias] = [None] * len(tables[table_for_alias].header)
            for pred in query.query_predicates_per_table[t_alias]:
                # info about the predicate, so the col name the sign and value
                pred_name = pred[0].strip().split('.')[1].strip()
                pred_sign = pred[1].strip()
                pred_value = pred[2].strip().replace("\'", "")

                col_index = tables[table_for_alias].header.index(pred_name)

                if pred_sign == '>=' or pred_sign == '>':
                    query_min_vals[t_alias][col_index] = pred_value
                elif pred_sign == '<=' or pred_sign == '<':
                    query_max_vals[t_alias][col_index] = pred_value
                elif pred_sign == '=':
                    query_min_vals[t_alias][col_index] = pred_value
                    query_max_vals[t_alias][col_index] = pred_value

            ar_model_query[t_alias] = [None] * (len(column_names_split[table_for_alias]) - 1)
            ar_model_query[t_alias] = [-1] + ar_model_query[t_alias]
            '''part for the exact estimator'''
            exact_query_min[t_alias] = query_min_vals[t_alias].copy()
            exact_query_max[t_alias] = query_max_vals[t_alias].copy()
            '''part for the grid index'''
            grid_index_min[t_alias] = query_min_vals[t_alias].copy()
            grid_index_max[t_alias] = query_max_vals[t_alias].copy()

            for col_indx, col_name in enumerate(column_names_split[table_for_alias][1:]):
                """
                    take the query value for the columns that are mapped and 
                    map it to the value used internally for the models
                """
                if exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]] is not None:
                    # for the min part
                    print(col_name.upper())
                    if col_name.upper() in tables[table_for_alias].column_mapper.keys():
                        print(tables[table_for_alias].column_mapper.keys())
                        print(tables[table_for_alias].column_mapper[col_name.upper()])
                        print(exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]])
                    elif col_name.lower() in tables[table_for_alias].column_mapper.keys():
                        print(tables[table_for_alias].column_mapper.keys())
                        print(tables[table_for_alias].column_mapper[col_name.lower()])
                        print(exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]])

                    if col_name.upper() in tables[table_for_alias].column_mapper.keys():
                        min_val_map = tables[table_for_alias].column_mapper[col_name.upper()][
                            exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]]]
                    elif col_name.lower() in tables[table_for_alias].column_mapper.keys():
                        if col_name == 'squawk':
                            min_val_map = tables[table_for_alias].column_mapper[col_name.lower()][
                                float(exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]])]
                        elif col_name == 'account':
                            """this is because pandas is treating the account number as an integer, so we have to convert it"""
                            min_val_map = tables[table_for_alias].column_mapper[col_name.lower()][
                                int(exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]])]
                        else:
                            # print(tables[table_for_alias].column_mapper[col_name.lower()])
                            min_val_map = tables[table_for_alias].column_mapper[col_name.lower()][
                                exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]]]

                    exact_query_min[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]] = min_val_map


                    # for the max part
                    exact_query_max[t_alias][ar_model_columns_indexes[table_for_alias][col_indx]] = min_val_map

                    # for the ar model need to put it on another position
                    ar_model_query[t_alias][col_indx + 1] = min_val_map


            """
                no need to store the dimensions that we will be ignoring for the creation of the grid 
            """
            for dim_to_delete in dims_to_delete_for_querying[table_for_alias]:
                del grid_index_min[t_alias][dim_to_delete]
                del grid_index_max[t_alias][dim_to_delete]


            for dim, _ in enumerate(grid_index_min[t_alias]):
                if grid_index_min[t_alias][dim] is None:
                    grid_index_min[t_alias][dim] = grid_indexes[table_for_alias].min_per_dimension[dim]
                    grid_index_max[t_alias][dim] = grid_indexes[table_for_alias].max_per_dimension[dim]
                else:
                    col_name = tables[table_for_alias].header[column_indexes_of_cols_not_in_ar_model[table_for_alias][dim]]
                    """need to check also for the grid index since we have dates now"""
                    if col_name.upper() in tables[table_for_alias].column_mapper.keys():
                        min_val_map = tables[table_for_alias].column_mapper[col_name.upper()][
                            grid_index_min[t_alias][dim]]

                        max_val_map = tables[table_for_alias].column_mapper[col_name.upper()][
                            grid_index_max[t_alias][dim]]

                        grid_index_min[t_alias][dim] = min_val_map
                        grid_index_max[t_alias][dim] = max_val_map
                    elif col_name.lower() in tables[table_for_alias].column_mapper.keys():
                        min_val_map = tables[table_for_alias].column_mapper[col_name.lower()][
                            grid_index_min[t_alias][dim]]

                        max_val_map = tables[table_for_alias].column_mapper[col_name.lower()][
                            grid_index_max[t_alias][dim]]

                        grid_index_min[t_alias][dim] = min_val_map
                        grid_index_max[t_alias][dim] = max_val_map
                    else:
                        """since we are reading strings from the csv file, transfer it to floats"""
                        grid_index_min[t_alias][dim] = float(grid_index_min[t_alias][dim])
                        grid_index_max[t_alias][dim] = float(grid_index_max[t_alias][dim])

        query.grid_index_query_min = grid_index_min
        query.grid_index_query_max = grid_index_max
        query.ar_model_queries = ar_model_query

        print('grid query min')
        print(query.grid_index_query_min)
        print('grid query max')
        print(query.grid_index_query_max)
        print('ar model query')
        print(query.ar_model_queries)


    print('ESTIMATING------------------------------------:')
    q_error = list()
    q_error_original = list()
    time_total = list()
    time_total_grid = list()
    time_total_join = list()
    time_total_exact = list()
    other_query_index = 0
    relative_error_range = list()
    id_of_range_object_of_interest = 0
    total_num_queries_checked = 0
    query_estimations = list()
    tmp_query_results = list()
    number_of_tables_per_query = list()
    for query_indx, query in enumerate(all_queries):
        if int(query.query_cardinality) == 0:
            other_query_index += 1
            continue

        total_num_queries_checked += 1

        print(f'working with query {query_indx + 1}')
        print(query.join_pairs)

        """estimating grid cells"""
        qualifying_cells_for_table = dict()
        total_time_qualifying_cells = 0
        time_range_joins = 0
        for t_alias in query.grid_index_query_max.keys():
            t_n = query.table_alias_to_table_map[t_alias]
            time_s_grid = time.time()
            # _ was estimated_cardinality, but since we don't need it except for printing , I'm not storing it
            qualifying_cells, _ = grid_indexes[t_n].range_join_qualifying_cells(query_min_vals=query.grid_index_query_min[t_alias],
                                                                                             query_max_vals=query.grid_index_query_max[t_alias],
                                                                                             values_ar_model=query.ar_model_queries[t_alias],
                                                                                             columns_for_ar_model=column_names_split[t_n])
            time_e_grid = (time.time() - time_s_grid) * 1000
            qualifying_cells_for_table[t_alias] = np.array(copy.deepcopy(qualifying_cells))
            total_time_qualifying_cells += time_e_grid

            print(f"There are {len(qualifying_cells)} qualifying cells for {t_n} as {t_alias} ! found for {time_e_grid}")

        print(f'Total time for {len(query.grid_index_query_max.keys())} tables for qualyfing cells is {total_time_qualifying_cells}')

        number_of_tables_per_query.append(len(query.grid_index_query_max.keys()))
        final_cells_result = dict()
        rps = dict()
        join_uniformity = 1
        signs_found_in_range_join_predicate = ['+', '-', '/', '*', ')']
        previous_table_of_interest = None
        time_update_range_predicates = 0
        for join_pair in query.join_pairs.keys():
            """
                For every join pair, need to know:
                 1. Which table will be used in the successive join pairs
                 3. Reorder the join pair in this regard
                 4. Consider only cells that have overlap and produce results per cell
                 5. Keep overlapping cells with the current table for the next one
            """
            time_update_range_predicate = time.time()
            rps[join_pair] = list()

            for join_condition in query.join_pairs[join_pair]:

                where_to_start_left = len(join_condition.left_table) + 1 # + 1 is for the dot
                where_to_start_right = len(join_condition.right_table) + 1 # + 1 is for the dot
                if 'sin(' in join_condition.left_side:
                    where_to_start_left = len(join_condition.left_table+'sin(') + 1 # + 1 is for the dot
                if 'sin(' in join_condition.right_side:
                    where_to_start_right = len(join_condition.right_table+'sin(') + 1 # + 1 is for the dot

                index_left = -1
                index_right = -1

                for math_oper in signs_found_in_range_join_predicate:

                    if math_oper in join_condition.left_side:
                        index_left = join_condition.left_side.index(math_oper)
                    if math_oper in join_condition.right_side:
                        index_right = join_condition.right_side.index(math_oper)

                if index_left == -1:
                    """this means that there was no sign so we take the end of the condition"""
                    index_left = len(join_condition.left_side)
                if index_right == -1:
                    """this means that there was no sign so we take the end of the condition"""
                    index_right = len(join_condition.right_side)

                lt_name = query.table_alias_to_table_map[join_condition.left_table]
                rt_name = query.table_alias_to_table_map[join_condition.right_table]

                # extract only the column name in the left part of the condition
                left_column = join_condition.left_side[where_to_start_left:index_left]
                left_column_index = column_names_of_cols_not_in_ar_model[lt_name].index(left_column)

                # extract only the column name in the right part of the condition
                right_column = join_condition.right_side[where_to_start_right:index_right]
                right_column_index = column_names_of_cols_not_in_ar_model[rt_name].index(right_column)

                if join_condition.right_table in join_condition.table_used_in_succcessive_joins.keys() or previous_table_of_interest == join_condition.right_table:

                    previous_table_of_interest = join_condition.right_table
                    """
                        By knowing which table is used in one of the successive join conditions we know how to execute
                        the estimation, i.e., over which cells to iterate and which cells to send as a list 
                        NEED TO SWAP THINGS 
                    """
                    if join_condition.sign == '>':
                        join_condition.sign = '<'
                    elif join_condition.sign == '<':
                        join_condition.sign = '>'
                    elif join_condition.sign == '<=':
                        join_condition.sign = '>='
                    elif join_condition.sign == '>=':
                        join_condition.sign = '<='

                    tmp_left_side = join_condition.left_side
                    join_condition.left_side = join_condition.right_side.replace(
                        join_condition.right_table + '.' + right_column, 'a')
                    join_condition.right_side = tmp_left_side.replace(
                        join_condition.left_table + '.' + left_column, 'b')

                    # change the tables now
                    tmp_switch_table = join_condition.left_table
                    join_condition.left_table = join_condition.right_table
                    join_condition.right_table = tmp_switch_table

                    tmp_index = left_column_index
                    left_column_index = right_column_index
                    right_column_index = tmp_index

                else:
                    join_condition.left_side = join_condition.left_side.replace(
                        join_condition.left_table + '.' + left_column, 'a')
                    join_condition.right_side = join_condition.right_side.replace(
                        join_condition.right_table + '.' + right_column, 'b')
                    previous_table_of_interest = join_condition.left_table

                tmp_rp = range_filter.RangeJoinPredicate(join_condition.sign, column_ids=[left_column_index, right_column_index],
                                                         expressions=[join_condition.left_side,
                                                                      join_condition.right_side]
                                                         )

                rps[join_pair].append(tmp_rp)

            time_e_update_range_predicate = (time.time() - time_update_range_predicate) * 1000
            time_update_range_predicates += time_e_update_range_predicate


            left_table_estimation = join_condition.left_table
            right_table_estimation = join_condition.right_table
            print(f'left table is {left_table_estimation}')
            """working with the current join pair"""

            time_s = time.time()
            buckets2_card = np.array([])
            for i, b in enumerate(qualifying_cells_for_table[right_table_estimation]):
                b.estimation_range_id = i
                buckets2_card = np.append(buckets2_card, b.num_points)

            list_bucket_2 = list()

            start_index_next_bucket_per_condition = list()

            is_join_attribute_same = True

            previous_join_attr = rps[join_pair][0].column_ids[0]
            for p_i, predicate in enumerate(rps[join_pair]):
                if predicate.operator == ">":
                    tmp_buckets2 = sorted(qualifying_cells_for_table[right_table_estimation], key=lambda x: (x.max_boundaries[predicate.column_ids[1]], x.min_boundaries[predicate.column_ids[1]])\
                        ,reverse=True)
                    start_index_next_bucket_per_condition.append(len(tmp_buckets2))

                else:
                    tmp_buckets2 = sorted(qualifying_cells_for_table[right_table_estimation], key=lambda x: (x.min_boundaries[predicate.column_ids[1]], x.max_boundaries[predicate.column_ids[1]]),
                                      reverse=False)
                    start_index_next_bucket_per_condition.append(0)


                list_bucket_2.append(tmp_buckets2)

                # this is to check if we can make use of the optimization with sorting
                # if the attribute is the same in all conditions then use the optimization, otherwise bucket by bucket
                if previous_join_attr != predicate.column_ids[0]:
                    is_join_attribute_same = False
                previous_join_attr = rps[join_pair][0].column_ids[0]


            res = 0

            res_buckets = list()
            tmp_results = dict()

            # sort original buckets just in some order
            left_table_buckets_sorted = sorted(qualifying_cells_for_table[left_table_estimation],
                                  key=lambda x: (x.min_boundaries[rps[join_pair][0].column_ids[0]], x.max_boundaries[rps[join_pair][0].column_ids[0]]),
                                  reverse=False)

            """parallel approach"""
            if is_join_attribute_same:
                """
                    if first and second attribute are the same, do the remembering of the buckets 
                """

                set_zeros = [[] for i in range(len(rps[join_pair]))]
                for bucket_indx, bucket1 in enumerate(left_table_buckets_sorted):
                    if is_join_attribute_same:
                        return_cell, cell_estimate, updated_cell_card, start_index_next_bucket_per_condition, set_zeros = range_filter. \
                                        check_range_parallel_multijoin_skipping_range(list_bucket_2, rps[join_pair], buckets2_card,
                                                                                    start_index_next_bucket_per_condition, set_zeros,
                                                                                    100, bucket1)

                    if return_cell is not None:
                        # need to update the number of points for the cell
                        # update the number of points to be multiplied with the cells of the next table
                        # we compute the cardinality estimate by multiplying the number of points for every cell with the overlap between the cells
                        # so always carry on the number of cells
                        return_cell.num_points = cell_estimate
                        res_buckets.append(return_cell)
                        res += math.ceil(cell_estimate)
                        tmp_results[return_cell.id] = math.ceil(cell_estimate)

            else:
                """
                    if the following attributes are not the same, execute things in parallel
                """
                max_workers = 3
                res_buckets, tmp_results = range_filter.parallel_execution_multijoin_already_sorted_input(max_workers,
                                                                                            left_table_buckets_sorted,
                                                                                            list_bucket_2,
                                                                                            rps[join_pair],
                                                                                            buckets2_card)

            """end parallel approach"""


            time_e = (time.time() - time_s) * 1000
            print(f'previous buckets for {left_table_estimation} were {len(qualifying_cells_for_table[left_table_estimation])}')
            qualifying_cells_for_table[left_table_estimation] = res_buckets
            print(f'now for {left_table_estimation} there are {len(qualifying_cells_for_table[left_table_estimation])}')
            final_cells_result = tmp_results
            print(f'result for this range join is {res}')
            print('current')
            print(tmp_results)
            join_uniformity *= res
            time_range_joins += time_e



        # take both the time from the grid and the time for the cells
        time_total.append(time_range_joins + (total_time_qualifying_cells) + time_update_range_predicates)
        time_total_grid.append(total_time_qualifying_cells)
        time_total_join.append(time_range_joins + time_update_range_predicates)

        # this is just using the existing result
        true_range_result = query.query_cardinality

        res = sum([final_cells_result[tmp_cell] for tmp_cell in final_cells_result.keys()])

        query_estimations.append(res)
        tmp_query_results.append(true_range_result)

        print(f'Estimated range cardinality {res} out of {true_range_result} q-error {max(res / true_range_result, true_range_result / res) if res != 0 and true_range_result != 0 else max(1. / true_range_result, true_range_result / 1.)} for {time_range_joins + (total_time_qualifying_cells) + time_update_range_predicates} MS')
        print(f'Estimated range cardinality with uniformity {join_uniformity} out of {true_range_result} for {time_range_joins + (total_time_qualifying_cells)} MS')

        q_error.append(max(res / true_range_result, true_range_result / res)
                       if res != 0 and true_range_result != 0 else max(1. / true_range_result, true_range_result / 1.))
        relative_error_range.append(abs((true_range_result - res) / res) if res !=0 else abs((true_range_result - 1.) / 1.))
        other_query_index += 1




    """PRINTING THE RESULTS"""
    print(f'Average q-error {np.average(q_error)}')
    print(f'Median q-error {np.quantile(q_error, .5)}')
    print(f'90 percentile q-error {np.quantile(q_error, .9)}')
    print(f'99 percentile q-error {np.quantile(q_error, .99)}')
    print(f'Max q-error {np.quantile(q_error, 1.)}')
    print(f'Average relative error {np.average(relative_error_range)}')
    print(f'Median relative error {np.quantile(relative_error_range, .5)}')
    print(f'90 percentile relative error {np.quantile(relative_error_range, .9)}')
    print(f'99 percentile relative error {np.quantile(relative_error_range, .99)}')
    print(f'Max relative error {np.quantile(relative_error_range, 1.)}')
    print(
        f'Our approach avg execution time: {np.average(time_total)} MS; grid {np.average(time_total_grid)} MS; join {np.average(time_total_join)} MS')
    print(
        f'Our approach median execution time: {np.quantile(time_total, .5)} MS; grid {np.quantile(time_total_grid, .5)} MS; join {np.quantile(time_total_join, .5)} MS')
    print(
        f'Our approach minimal execution time: {np.quantile(time_total, .0)} MS; grid {np.quantile(time_total_grid, .0)} MS; join {np.quantile(time_total_join, .0)} MS')
    print(
        f'Our approach maximal execution time: {np.quantile(time_total, 1.)} MS; grid {np.quantile(time_total_grid, 1.)} MS; join {np.quantile(time_total_join, 1.)} MS')

    print(f'Total number of queries checked {total_num_queries_checked}')


    """uncomment to store results"""
    # with open('range_join_results/customer{}_{}_size_2_inequality_queries.txt'.format('cdf_5_5_5_residual', 'range'), 'w') as store_file:
    #     store_file.write('time,num_tables,true_result,estimated_result\n')
    #     for query_indx, time_q in enumerate(time_total):
    #         store_file.write(str(time_q) + ',' + str(number_of_tables_per_query[query_indx]) + ',' + str(tmp_query_results[query_indx]) + ',' + str(
    #             query_estimations[query_indx]) + '\n')
    """end uncomment to store results"""




