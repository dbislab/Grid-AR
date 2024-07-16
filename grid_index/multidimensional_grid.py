import math

import numpy as np
import time
import pickle
import glob

import data_location_variables
from .grid_cell import GridCell
from .range_filter import GridCellResult
from .range_regressor import RangeRegressor
import train_model
from estimators import BaseDistributionEstmationBatch
import torch

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

class MultidimensionalGrid:

    def __init__(self, data, num_ranges, cdf_based=False, dimensions_to_ignore=None, column_indxs_for_ar_model_training=[], column_names='',
                 table_name='', table_size=None):
        # take the number of ranges in which every dimension will be split
        self.num_ranges_per_dimension = np.array(num_ranges)
        self.cdf_based = cdf_based
        self.num_dimensions = len(data)
        # # these are dimensions that we do not want to have for the creation of the grid
        self.dimensions_to_ignore = dimensions_to_ignore
        # # these are only the dimensions that we want to have for the creation of the grid
        self.relevant_dimensions = [dim for dim in range(self.num_dimensions) if dim not in self.dimensions_to_ignore]
        # these are the column indexes to extract the values that will be used for training the AR model
        # give them in the order that we want the AR model to be trained
        self.column_indxs_for_ar_model_training = column_indxs_for_ar_model_training
        # the table size needed for the ar model
        self.table_size = table_size
        # the ar model for estimation
        self.estimator = None

        print('we have this many dimensions and this many ranges sent:')
        print(self.num_dimensions)
        print(len(self.num_ranges_per_dimension))
        print()
        print('what to train ar model on')
        print(self.column_indxs_for_ar_model_training)

        """
            The cell multipliers needed for computing a cell for a data point
        """
        self.dims_multipliers = self.compute_cell_multipliers_per_dimension(self.num_dimensions)
        print('created the following dimension multipliers')
        print(self.dims_multipliers)
        print()

        if self.cdf_based:
            """
                Do the creation of ranges and cells CDF-based.
            """
            # take the maximum and minimum per dimension
            self.max_per_dimension, self.min_per_dimension = self.get_max_and_min_per_dimension(data)

            self.dimension_regressors = list()
            dim_of_interest = 0
            """compute the cdf per range"""
            for dim_id, dimension_data in enumerate(data):
                if dim_id in self.relevant_dimensions:
                    dim_regressor = RangeRegressor(dimension_data)

                    self.dimension_regressors.append(dim_regressor)

            print(self.num_ranges_per_dimension)

            self.cell_ids_per_dimension_range = dict()
            for dim, num_ranges in enumerate(self.num_ranges_per_dimension):
                self.cell_ids_per_dimension_range[dim] = dict()
                for range_id in range(num_ranges + 1):
                    self.cell_ids_per_dimension_range[dim][range_id] = set()
        else:
            """
                Do the creation of ranges and cells based on ranges specified by MAX and MIN.
            """
            # take the maximum and minimum per dimension
            self.max_per_dimension, self.min_per_dimension = self.get_max_and_min_per_dimension(data)
            """
               Create ranges for the dimensions
            """
            self.range_objects_per_dimension = self.create_range_objects_for_dimensions()
            print('created the following ranges per dimension')
            print(self.range_objects_per_dimension)
            print()

            self.cell_ids_per_dimension_range = dict()
            for dim, num_ranges in enumerate(self.num_ranges_per_dimension):
                self.cell_ids_per_dimension_range[dim] = dict()
                for range_id in range(num_ranges + 1):
                    self.cell_ids_per_dimension_range[dim][range_id] = set()

        """
            Map the data to cells
        """
        print("We are starting to insert")
        self.data_to_cells_map = self.data_to_cells_mapping(data, column_names=column_names, table_name=table_name)

    def compute_cell_multipliers_per_dimension(self, num_dimensions):
        dims_multipliers = np.array([])
        for dim in range(len(self.num_ranges_per_dimension)):
            """
                calculate the cell multipliers so that we can represent the cell as an integer at the end
                represented as:
                    num_ranges ^ (num_dimensions - (dimension + 1))
                so for every dimension, is the total number of ranges for the dimension on the power 
                of the number of dimensions minus the current dimension index plus 1.
            """
            dims_multipliers = np.append(dims_multipliers, int(math.pow(self.num_ranges_per_dimension[dim], num_dimensions - (dim + 1))))
        return dims_multipliers

    def create_range_objects_for_dimensions(self):
        """
        For every dimension create the ranges as objects and store the
        respective points per range.
        :param num_dimensions: the number of dimensions for the data
        :param data: the data
        :return: ranges as objects per dimension
        """
        range_objects_per_dimension = self.create_ranges()

        return range_objects_per_dimension

    def create_ranges(self):
        """
        Creating ranges per dimension as range objects.
        The start boundary is inclusive in EVERY range.
        For every range, the ending value IS NOT inclusive in the range.
        :param num_dimensions: the number of dimensions in the data
        :return: ranges per dimension such that start value is INCLUSIVE and end value is NOT
        """
        ranges_per_dimension = dict()

        for dimension in range(len(self.num_ranges_per_dimension)):
            # get the range length for the current dimension
            range_length = self.get_range_length(dimension)
            # ranges for the current dimension
            current_ranges = self.create_ranges_for_dimension(dimension, range_length)

            ranges_per_dimension[dimension] = current_ranges

        return ranges_per_dimension

    def create_ranges_for_dimension(self, dimension, range_length):
        """
        Computes the ranges for the given dimension.
        The start boundary is INCLSUIVE in the range.
        The end boundary for every range is NOT part of the range.
        :param dimension: the dimension of interest
        :param range_length: the range length
        :return: returns self.num_ranges dimensions of length range_length
        """
        ranges = dict()
        for range_id in range(self.num_ranges_per_dimension[dimension]):
            range_start = self.get_range_start(range_id, range_length, self.min_per_dimension[dimension])
            range_end = self.get_range_end(range_start, range_length)
            ranges[range_id] = (range_start, range_end)
        return ranges

    def get_range_length(self, dimension):
        """
        For a given dimension, it calculates the length of the range.
        :param dimension: the dimension of interest
        :return: the range length for that dimension
        """
        return (self.max_per_dimension[dimension] - self.min_per_dimension[dimension] + 1) / self.num_ranges_per_dimension[dimension]

    def get_cell_dimension_divisor(self, dimension):
        """
        Parameter needed for the computation of the cell.
        Computed as: MAX_d - MIN_d + 1. So the maximum for dimension d minus the minimum for the same dimension.
        :param dimension: the dimension for which we are working
        :return: the divisior for the cell computation
        """
        return self.max_per_dimension[dimension] - self.min_per_dimension[dimension] + 1

    def get_max_and_min_per_dimension(self, data):
        """
        Calculates the maximum and minimum value for every dimension.
        :param data: the data per dimension as list of lists
        :return: max_per_dimension AND min_per_dimension
        """
        max_per_dimension = np.array([])
        min_per_dimension = np.array([])

        for dimension in range(self.num_dimensions):
            if dimension in self.relevant_dimensions:
                data_dimension = data[dimension]
                max_per_dimension = np.append(max_per_dimension, max(data_dimension))
                min_per_dimension = np.append(min_per_dimension, min(data_dimension))
        return max_per_dimension, min_per_dimension

    def get_range_index_for_data_point(self, x, dimension):
        """
        For a data point and dimension, get the range to which this data point should belong.
        :param x: the data point of interest
        :param dimension: the dimension for which the check is performed
        :return: the range id to which this data point belongs
        """
        if x < self.min_per_dimension[dimension]:
            x = self.min_per_dimension[dimension]
            """If the value is outside of the dimension values return -1"""
            # return -1
        elif x > self.max_per_dimension[dimension]:
            x = self.max_per_dimension[dimension]
            """If the value is outside of the dimension values return -1"""

        return int(math.floor((x - self.min_per_dimension[dimension]) / self.get_range_length(dimension)))

    def get_range_start(self, i, range_length, min):
        """
        Computes the start boundary for the range i.
        The start boundary is INCLUSIVE in the range.
        :param i: the index of the range
        :param range_length: the length of the ranges for the dimension
        :param min: the minimal value for the dimension
        :return: the start boundary for range i for the dimension of interest.
        """
        return range_length * i + min

    def get_range_end(self, range_start, range_length):
        """
        Computes the end boundary for a range.
        :param range_start: The start boundary of the range of interest.
        :param range_length: the length of the range for the dimension of ineterst
        :return: the end range boundary EXCLUSIVE
        """
        return range_start + range_length

    def get_ranges_for_equality_two_points_cdf(self, point1, point2):
        """
        Calculate the range indexes per dimension for 2 points at a time.
        Use the CDF for calculating the range indexes.
        :param point1: first point
        :param point2: second point
        :return: list of range indexes per point
        """
        ranges_point1 = list()
        ranges_point2 = list()
        for dimension in range(len(point1)):
            # take the cdf value for the tuple value of dimension dim
            range_point1, range_point2 = self.get_cdf_for_tuples(dimension, [[point1[dimension]], [point2[dimension]]])
            # compute the range by multiplying with the number of ranges and taking the floor
            ranges_point1.append(math.floor(range_point1 * self.num_ranges_per_dimension[dimension]))
            ranges_point2.append(math.floor(range_point2 * self.num_ranges_per_dimension[dimension]))

        return ranges_point1, ranges_point2

    def get_ranges_for_equality_two_points(self, point1, point2):
        ranges_point1 = list()
        ranges_point2 = list()

        for dimension in range(len(self.num_ranges_per_dimension)):
            ranges_point1.append(self.get_range_index_for_data_point(point1[dimension], dimension))
            ranges_point2.append(self.get_range_index_for_data_point(point2[dimension], dimension))

        return ranges_point1, ranges_point2

    def get_cdf_for_tuples(self, dimension, values):
        """
        Takes as input the dimension and the list of values for which we want to
        predict the CDF.
        :param dimension: the dimension of interest
        :param values: list of n values of the form [[val_1], ..., [val_n]]
        :return: n CDFs one for each input value
        """
        return self.dimension_regressors[dimension].predict_cdf(values)

    def get_range_for_tuple_cdf(self, dimension, dim_value):
        """
        For a given value and dimension, compute the CDF and based on it compute the range
        where the value belongs. The range index is computed as:
            math.FLOOR(CDF(dim_value) * num_ranges)
        :param dimension: the dimension of interest
        :param dim_value: the value for which we are computing CDF
        :return: the range index for the value 'dim_value'
        """
        return math.floor(self.dimension_regressors[dimension].predict_cdf([[dim_value]]) * self.num_ranges_per_dimension[dimension])

    def get_cell_id_for_tuple_cdf(self, data_tuple, data_cdf_val_per_dim=None, tuple_id=None):
        """
        Computes the cell id for a tuple based on the range ids.
        For computing the range ids it uses the CDFs.
        :param data_tuple: the tuple of interest
        :return: the cell index where the tuple belongs
        """
        cell_value_as_integer = 0
        cell_values = ""

        qualifying_ranges_per_dimension = list()
        for dim, val_dim in enumerate(data_tuple):
            """
                use the same trick to compute the cell id, i.e.,
                multiply the range for every dimension with a special value so that we get the
                right ordering of the cells.
            """
            if data_cdf_val_per_dim is not None:
                # already precomputed this, make use of the cdf value, just transfer it to the right range
                dimension_range = math.floor(data_cdf_val_per_dim[dim][tuple_id] * self.num_ranges_per_dimension[dim])
            else:
                dimension_range = self.get_range_for_tuple_cdf(dim, val_dim)

            # need to store all the range ids to be able to map cell ids to them later on
            qualifying_ranges_per_dimension.append(dimension_range)

            cell_value_as_integer += dimension_range * self.dims_multipliers[dim]
            cell_values = cell_values + str(dimension_range)

        # map the cell id to all the respective range ids
        for dim, range_index in enumerate(qualifying_ranges_per_dimension):
            if range_index == None:
                # if none, then we should be ignoring the ranges for this dimensions
                continue

            self.cell_ids_per_dimension_range[dim][range_index].add(cell_value_as_integer)


        return cell_value_as_integer

    def get_cell_id_for_tuple(self, data_tuple):
        cell_value_as_integer = 0

        qualifying_ranges_per_dimension = list()

        for dim, val_dim in enumerate(data_tuple):
            """
                the value for the cell for dimension dim is calculated as:
                    ((value_d - MIN_d) / cell_dim_divisor) * num_ranges

                So the value of the tuple in dimension d minus the minimum value for dimension d divided by the 
                cell divisor and multiplied with the number of ranges that we have per dimension. 
            """
            cell_dim_value = math.floor(((val_dim - self.min_per_dimension[dim]) / self.get_cell_dimension_divisor(dim))
                                        * self.num_ranges_per_dimension[dim])

            qualifying_ranges_per_dimension.append(cell_dim_value)

            # the cell representation for the data point as integer
            cell_value_as_integer = cell_value_as_integer + self.dims_multipliers[dim] * cell_dim_value
        """
            for every range value , store the cell id that we have computed 
        """

        for dim, range_index in enumerate(qualifying_ranges_per_dimension):
            if range_index == None:
                # if none, then we should be ignoring the ranges for this dimensions
                continue
            self.cell_ids_per_dimension_range[dim][range_index].add(cell_value_as_integer)


        return cell_value_as_integer

    def get_cell_id_from_range_ids(self, range_ids_per_dimension):
        cell_value = 0
        for dim, range_id in enumerate(range_ids_per_dimension):
            cell_value = cell_value + self.dims_multipliers[dim] * range_id
        return cell_value

    def all_cells_computed_from_ranges(self, tuple_min, tuple_max):
        """
        :param tuple_min:
        :param tuple_max:
        :return:
        """
        if self.cdf_based:
            # computes the qualifying ranges per dimension for the minimal and maximal tuple
            # at the same time to speed up things. The ranges are computed based on CDFs
            ranges_min_tuple, ranges_max_tuple = self.get_ranges_for_equality_two_points_cdf(tuple_min, tuple_max)
        else:
            # computes the qualifying ranges per dimension for the minimal and maximal tuple
            # at the same time to speed up things
            ranges_min_tuple, ranges_max_tuple = self.get_ranges_for_equality_two_points(tuple_min, tuple_max)

        all_qualifying_cells = set()
        first_dim = True
        for dim in range(len(self.num_ranges_per_dimension)):
            starting_range_for_dimension = ranges_min_tuple[dim]
            ending_range_for_dimension = ranges_max_tuple[dim]
            cells_per_dimension = list()
            while starting_range_for_dimension <= ending_range_for_dimension:
                cells_per_dimension.extend(self.cell_ids_per_dimension_range[dim]
                                                                [starting_range_for_dimension])
                starting_range_for_dimension += 1

            if len(all_qualifying_cells) > 0:
                all_qualifying_cells = all_qualifying_cells.intersection(set(cells_per_dimension))
            elif first_dim:
                all_qualifying_cells = set(cells_per_dimension)
                first_dim = False
            else:
                break

        return all_qualifying_cells

    def all_cells_for_range_query(self, tuple_min, tuple_max):
        """
        As input receives the minimal and maximal tuple for the query and
        then it identifies all the cell ids between these maximal and minimal index.
        However, it returns all cell ids even those of cells that might not be suitable
        for the query.
        :param tuple_min: tuple representing the minimal points of the query
        :param tuple_max: tuple representing the maximal points of the query
        :return: index of all the qualifying cells
        """
        if self.cdf_based:
            # computes the qualifying ranges per dimension for the minimal and maximal tuple
            # at the same time to speed up things. The ranges are computed based on CDFs
            ranges_min_tuple, ranges_max_tuple = self.get_ranges_for_equality_two_points_cdf(tuple_min, tuple_max)
        else:
            # computes the qualifying ranges per dimension for the minimal and maximal tuple
            # at the same time to speed up things
            ranges_min_tuple, ranges_max_tuple = self.get_ranges_for_equality_two_points(tuple_min, tuple_max)

        # get the cell id for the query in the minimal tuple
        cell_id_min = self.get_cell_id_from_range_ids(ranges_min_tuple)
        # get the cell id for the maximal tuple
        cell_id_max = self.get_cell_id_from_range_ids(ranges_max_tuple)

        all_cell_ids = set()
        # only take the cells that we are storing not the empty ones
        while cell_id_min <= cell_id_max:
            if cell_id_min in self.data_to_cells_map:
                all_cell_ids.add(cell_id_min)
            cell_id_min += 1

        return all_cell_ids


    def data_to_cells_mapping(self, data, column_names='', table_name=''):
        """
        Create a hash map where as keys we have the cell ids and as values
        all the data points that belong to that cell.
        :param data: the data represented as values per dimension
        :return: hash map with items:
            <cell_id1, [tuple1, tuple2, ...]>
            ...
            <cell_idn, [tuple1, tuple2, ...]>
        """
        data_cdf_val_per_dim = None
        if self.cdf_based:
            data_cdf_val_per_dim = list()
            dims_tmp_indx = 0
            for dim, dim_values in enumerate(data):
                if dim not in self.relevant_dimensions:
                    continue
                dim_values = np.array(dim_values).reshape(-1, 1)
                dim_predicted_cdfs = self.get_cdf_for_tuples(dims_tmp_indx, dim_values)
                data_cdf_val_per_dim.append(dim_predicted_cdfs)
                dims_tmp_indx += 1
            print(f'Predicted the cdf for {len(data_cdf_val_per_dim)} columns out of {len(data)} columns')

        # this is the data that will be given to the autoregressive model for training
        data_for_ar = list()
        # mapping of the grid cells to reduce the huge numbers
        grid_cell_to_int_mapping = dict()
        grid_cell_tmp_indx = 0

        data_map = dict()
        data_formatted = list(zip(*data))
        for tuple_id, data_tuple in enumerate(data_formatted):
            modified_tuple = [tuple_val for dim_indx, tuple_val in enumerate(data_tuple) if dim_indx not in self.dimensions_to_ignore]
            if self.cdf_based:
                # first calculate the cell_index for the data point
                # but based on CDFs per range
                cell_index = self.get_cell_id_for_tuple_cdf(modified_tuple, data_cdf_val_per_dim, tuple_id)
            else:
                # first calculate the cell_index for the data point
                cell_index = self.get_cell_id_for_tuple(modified_tuple)

            # check if the cell has already been created, if not create it
            if cell_index in data_map.keys():
                grid_cell = data_map[cell_index]
            else:
                grid_cell = GridCell(len(self.relevant_dimensions))
                grid_cell_to_int_mapping[cell_index] = grid_cell_tmp_indx
                grid_cell_tmp_indx += 1

            if data_location_variables.use_ar_model and len(self.column_indxs_for_ar_model_training) > 0:
                # instead of storing the actual index for the grid cell, store a mapped value to reduce the enormous values
                tuple_for_ar_model = [grid_cell_to_int_mapping[cell_index]]
                for indx_rel_col in self.column_indxs_for_ar_model_training:
                    """
                        extract only the values for the columns that will be used for training the ar model
                    """
                    tuple_for_ar_model.append(data_tuple[indx_rel_col])
                data_for_ar.append(tuple_for_ar_model)

            # add the point to the grid cell
            grid_cell.add_tuple(modified_tuple)

            # add the grid cell to the map
            data_map[cell_index] = grid_cell
            if tuple_id % 100000 == 0:
                print(f'covered {tuple_id} tuples')

        """
            Training an autoregressive model that will be used for generating samples per 
            join pair later on. To configure the size of the model, compression etc, go in train_model.py.
            Required:
                t_spec_name: the name of the table for which the ar model is created which will be how it is stored
                data: the actual data on which the model will be trained
                column_names: the names of the columns in the data
                dataset_name: it will be always 'table_special' to trigger the special training with custom dataset 
                              passed as input 
        """
        if data_location_variables.use_ar_model and len(data_for_ar) > 0:
            table_name = table_name + data_location_variables.grid_ar_name
            """check if model exists"""
            folder_name = 'models_vs_ar_modelps'
            path_to_model = './{}/{}-*.pt'.format(folder_name, table_name)
            all_names_in_folder = glob.glob(path_to_model)

            if len(all_names_in_folder) > 0:
                # the mapping of grid vals to int has to exist so read it
                with open('./column_mappings_grid/{}_grid_cells.pickle'.format(table_name), 'rb') as store_path:
                    grid_cell_to_int_mapping = pickle.load(store_path)

                """this means a model for the table exists and we would like to load it"""
                ar_model = train_model.LoadExistingModel(table_name, column_names, folder_name)
                """
                    need to read the compressor element also although it can be created every time for the table,
                    to avoid that just read it 
                """
                with open('./compressor_elems/{}_compressor.pickle'.format(table_name), 'rb') as handle:
                    ar_model.compressor_element = pickle.load(handle)
            else:
                """STORE THE GRID VALS TO INT MAPPING"""
                with open('./column_mappings_grid/{}_grid_cells.pickle'.format(table_name), 'wb') as store_path:
                    pickle.dump(grid_cell_to_int_mapping, store_path, protocol=pickle.HIGHEST_PROTOCOL)

                ar_model = train_model.TrainTask(t_spec_name=table_name, data=data_for_ar,
                                                  column_names=column_names, dataset_name='table_special',
                                                 save_folder_name=folder_name,
                                                 save_compressor_folder_name='compressor_elems')

            print(f'TRAINED AR MODEL for {table_name}')

            # build a base distribution estimator that will be used later on
            self.estimator = BaseDistributionEstmationBatch(ar_model, None, 1, device=DEVICE, shortcircuit=True,
                                                            num_sample=100000, cardinality=len(data_for_ar),
                                                            mapping_for_grid_cell=grid_cell_to_int_mapping)
        print('finished creating the grid now computing some statistics per cell')

        min_num_tuples_for_cell = 1000
        max_num_tuples_for_cell = 0
        sum_of_tuples = 0
        # sort all the points from the grid cell
        for cell_index in data_map.keys():
            cell_num_tuples = len(data_map[cell_index].cell_data)
            sum_of_tuples += cell_num_tuples
            if cell_num_tuples < min_num_tuples_for_cell:
                min_num_tuples_for_cell = cell_num_tuples
            if cell_num_tuples > max_num_tuples_for_cell:
                max_num_tuples_for_cell = cell_num_tuples

            data_map[cell_index].post_process_cell()


        print(f'The grid has {len(data_map.keys())} cells with some tuples.')
        print(f'The cell with least amount of tuples has {min_num_tuples_for_cell} tuples.')
        print(f'The cell with most amount of tuples has {max_num_tuples_for_cell} tuples.')
        print(f'The avg number of tuples for the grid is {sum_of_tuples / len(data_map.keys())}.')

        return data_map

    def range_join_qualifying_cells(self, query_min_vals, query_max_vals, values_ar_model=None, columns_for_ar_model=None):
        # all the qualifying cells for the query
        qualifying_cells = self.all_cells_computed_from_ranges(query_min_vals, query_max_vals)

        signs_ar_model = []
        for val_ar in values_ar_model:
            signs_ar_model.append('=' if val_ar is not None else None)

        signs_ar_model_batch = signs_ar_model * len(qualifying_cells)

        values_ar_model_batch = []
        estimate_per_cell = list()
        estimates_without_ar_model = list()
        range_query_cells = list()
        for cell_index in qualifying_cells:
            """create a cell object for the range join use case"""
            cell_obj = self.data_to_cells_map[cell_index]
            tmp_obj = GridCellResult(cell_index, cell_obj.min_val_per_dim, cell_obj.max_val_per_dim)
            range_query_cells.append(tmp_obj)
            """end create a cell object for the range join use case"""

            if all(v is None for v in values_ar_model[1:]):
                """if the query is not for the ar model, just do the estimate without the ar model """
                estimates_without_ar_model.append(np.ceil(self.data_to_cells_map[cell_index].
                                                          single_table_estimation_no_armodel(query_min_vals,
                                                                                             query_max_vals)[0]))
            else:
                """
                    if the query is for the ar model, do the estimation as the relation of the estimation with the num
                    of points and plus build the query for the ar model 
                """
                values_ar_model[0] = int(self.estimator.mapping_for_grid_cell[cell_index])

                values_ar_model_batch.append(list(values_ar_model))

                tmp_estimate, tmp_num_points = self.data_to_cells_map[cell_index]. \
                    single_table_estimation_no_armodel(query_min_vals, query_max_vals)
                estimate_per_cell.append(tmp_estimate / tmp_num_points)

        if len(estimate_per_cell) > 0 and len(estimates_without_ar_model) > 0:
            print('we got both estimates!!!!!')
            exit(1)

        query_cardinalities = list()
        query_cardinality = 0
        if len(values_ar_model_batch) > 0:

            results = self.estimator.QueryWithCompressionBatch(columns_for_ar_model, signs_ar_model_batch,
                                                               values_ar_model_batch,
                                                               num_data_in_batch=len(values_ar_model_batch))

            # since the results are an array of arrays, always take only the value to make it an array of values like
            # how estimate_per_cell is defined

            query_cardinalities = torch.ceil(torch.mul(results[:, 0], torch.tensor(estimate_per_cell, device=DEVICE)))
            query_cardinality = torch.sum(query_cardinalities)

        else:
            query_cardinalities = estimates_without_ar_model
            query_cardinality = sum(query_cardinalities)

        if torch.is_tensor(query_cardinalities):
            for est_indx, cell_query_estimate in enumerate(query_cardinalities):
                range_query_cells[est_indx].num_points = cell_query_estimate.item()
        else:
            for est_indx, cell_query_estimate in enumerate(query_cardinalities):
                range_query_cells[est_indx].num_points = cell_query_estimate

        return range_query_cells, query_cardinality


    def single_table_estimation(self, query_min_vals, query_max_vals, values_ar_model=None, columns_for_ar_model=None):
        start_time = time.time()

        qualifying_cells = self.all_cells_computed_from_ranges(query_min_vals, query_max_vals)

        query_cardinality = 0
        signs_ar_model = []
        for val_ar in values_ar_model:
            signs_ar_model.append('=' if val_ar is not None else None)

        signs_ar_model_batch = signs_ar_model * len(qualifying_cells)

        values_ar_model_batch = []
        estimate_per_cell = list()
        estimates_without_ar_model = list()

        if all(v is None for v in values_ar_model[1:]):
            for cell_index in qualifying_cells:
                """if the query is not for the ar model, just do the estimate without the ar model """
                estimates_without_ar_model.append(np.ceil(self.data_to_cells_map[cell_index].
                                                  single_table_estimation_no_armodel(query_min_vals, query_max_vals)[0]))
        else:
            for cell_index in qualifying_cells:
                """
                    if the query is for the ar model, do the estimation as the relation of the estimation with the num
                    of points and plus build the query for the ar model 
                """
                values_ar_model[0] = int(self.estimator.mapping_for_grid_cell[cell_index])

                values_ar_model_batch.append(list(values_ar_model))

                tmp_estimate, tmp_num_points = self.data_to_cells_map[cell_index].\
                    single_table_estimation_no_armodel(query_min_vals, query_max_vals)
                estimate_per_cell.append(tmp_estimate / tmp_num_points)

        if len(values_ar_model_batch) > 0:
            results = self.estimator.QueryWithCompressionBatch(columns_for_ar_model, signs_ar_model_batch, values_ar_model_batch,
                                                          num_data_in_batch=len(values_ar_model_batch))

            # since the results are an array of arrays, always take only the value to make it an array of values like
            # how estimate_per_cell is defined
            query_cardinality = torch.sum(torch.ceil(torch.mul(results[:, 0], torch.tensor(estimate_per_cell, device=DEVICE))))
        else:
            query_cardinality += sum(estimates_without_ar_model)
        end_time = (time.time() - start_time) * 1000

        return query_cardinality if query_cardinality > 0 else 1, end_time