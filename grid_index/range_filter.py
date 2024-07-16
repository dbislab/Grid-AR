import time
import numpy as np
from scipy import integrate
import numexpr as ne
from multiprocessing import Pool
from functools import partial
import math


class RangeJoinPredicate:
    def __init__(self, operator, column_ids, expressions = None, range_first_col=None, range_second_col=None, num_cells_first_col=None, num_cells_second_col=None):
        """
        The class containing the Range Join Predicates between the tables
        :param operator: Possible operators are [<, >, >=, <=]
        :param column_ids: Ids of the columns involved in the predicate
        :param expressions: Expressions that will shift the boundaries of the buckets, example x1 * 3 + 2 TODO see how to do this
        """
        self.operator = operator
        self.column_ids = column_ids
        self.expressions = expressions
        self.type = 0
        self.range_first_col = range_first_col
        self.range_second_col = range_second_col
        self.num_cells_first_col = num_cells_first_col
        self.num_cells_second_col = num_cells_second_col
        self.indexes_right_table = dict()

        if range_first_col is not None:
            a = range_first_col
            self.updated_expressions_first_col = ne.evaluate(self.expressions[0])

            b = range_second_col
            self.updated_expressions_second_col = ne.evaluate(self.expressions[1])


    def compute_overlap(self):
        common_area = min(self.updated_expressions_first_col[1], self.updated_expressions_second_col[1]) - \
                      max(self.updated_expressions_first_col[0], self.updated_expressions_second_col[0])


        whole_area = max(self.updated_expressions_first_col[1], self.updated_expressions_second_col[1]) - \
                     min(self.updated_expressions_first_col[0], self.updated_expressions_second_col[0])


        return common_area / whole_area

    def __eq__(self, other):
        """
        Checking if the first is greater than the second. If the second has more
        :param other:
        :return:
        """
        if not isinstance(other, RangeJoinPredicate):
            raise TypeError('can only compare two RangeJoinPredicates')

        if self.compute_overlap() == other.compute_overlap():
            return True
        return False

    def __gt__(self, other):
        """
        Checking if the first is greater than the second. If the second has more
        :param other:
        :return:
        """
        if not isinstance(other, RangeJoinPredicate):
            raise TypeError('can only compare two RangeJoinPredicates')

        if self.compute_overlap() < other.compute_overlap():
            return False

        return True

    def __lt__(self, other):
        """
        Checking if the first is lower than the second.
        :param other:
        :return:
        """
        if not isinstance(other, RangeJoinPredicate):
            raise TypeError('can only compare two RangeJoinPredicates')

        if self.compute_overlap() < other.compute_overlap():
            return True

        return False

class GridCellResult:
    def __init__(self, id, min_boundaries, max_boundaries, est_cell_cardinality=None, per_dimension_distribution=None):
        self.id = id
        self.min_boundaries = np.array(min_boundaries)
        self.max_boundaries = np.array(max_boundaries)
        self.num_points = est_cell_cardinality
        self.per_dimension_distribution = per_dimension_distribution
        bounding_box_volume_data = (self.max_boundaries - self.min_boundaries)
        self.bounding_box_volume = np.product([i for i in bounding_box_volume_data if i != 0])
        self.estimation_range_id = -1

    def __str__(self):
        return str(self.id) + ", " + str(self.min_boundaries) + ", " + str(self.max_boundaries) + ": " + str(self.num_points)

    def compute_volume(self, query):
        range_start = np.max([query[0], self.min_boundaries], axis=0)
        range_end = np.min([query[1], self.max_boundaries], axis=0)
        volume_overlap = 1.0
        for i in range(len(range_start)):
            if range_end[i] < range_start[i]:
                return 0
            else:
                diff = (range_end[i] - range_start[i])
                if diff != 0:
                    volume_overlap *= diff
        return volume_overlap / self.bounding_box_volume

def check_range_parallel_multijoin_skipping_range(list_buckets_2, range_join_predicates, buckets2_card, start_index_next_bucket_per_condition, set_zeros, num_samples, bucket1):
    total_card = 0
    # boundaries_min_deepcopy = copy.deepcopy(bucket1.min_boundaries)
    # boundaries_max_deepcopy = copy.deepcopy(bucket1.max_boundaries)
    overlap = np.ones(len(list_buckets_2[0])) * bucket1.num_points

    for p_i, predicate in enumerate(range_join_predicates):
        a = bucket1.min_boundaries[predicate.column_ids[0]]
        min_x = ne.evaluate(predicate.expressions[0])
        a = bucket1.max_boundaries[predicate.column_ids[0]]
        max_x = ne.evaluate(predicate.expressions[0])


        # do not take all the buckets for the second list
        # i.e., ignore all buckets for which the previous bucket1 was completely greater
        if predicate.operator == "<":
            buckets2 = list_buckets_2[p_i][start_index_next_bucket_per_condition[p_i]:len(list_buckets_2[p_i])]

            tmp_start = start_index_next_bucket_per_condition[p_i]

            if len(buckets2) == 0:
                """if there are no buckets to check for the smaller condition then everything is definitely 0"""
                return None, total_card, None, start_index_next_bucket_per_condition, set_zeros

            if len(set_zeros[p_i]) == len(overlap):
                overlap = np.zeros(len(list_buckets_2[p_i])) * bucket1.num_points
            elif len(set_zeros[p_i]) > 0:
                overlap.put(set_zeros[p_i], 0.)
        else:
            buckets2 = list_buckets_2[p_i][0:start_index_next_bucket_per_condition[p_i]]

        """
            Because of the min based ordering for <, it can happen that the max boundary in the next range 
            is smaller than the current (previous) one. Thus, avoid skipping everything in between. 
        """
        overlapping_buckets_inbetween_exist = False

        for j, rect_j in enumerate(buckets2):
            if overlap[rect_j.estimation_range_id] == 0:
                continue

            b = rect_j.min_boundaries[predicate.column_ids[1]]
            min_y = ne.evaluate(predicate.expressions[1])
            b = rect_j.max_boundaries[predicate.column_ids[1]]
            max_y = ne.evaluate(predicate.expressions[1])


            if predicate.operator == "<":
                if max_x <= min_y:
                    break
                elif min_x >= max_y:
                    overlap[rect_j.estimation_range_id] = 0
                    set_zeros[p_i].append(rect_j.estimation_range_id)
                    if not overlapping_buckets_inbetween_exist:
                        start_index_next_bucket_per_condition[p_i] = tmp_start + j + 1
                else:
                    """if next range has a smaller max, we know not to jump the inbetween ranges that have overlap"""
                    overlapping_buckets_inbetween_exist = True
                    overlap[rect_j.estimation_range_id] *= overlap_calculation(min_x, max_x, min_y, max_y, num_samples=num_samples)
                    """updating bounds"""
                    # if max_y < boundaries_max_deepcopy[predicate.column_ids[0]]:
                    #     # boundaries_max_deepcopy[predicate.column_ids[0]] = max_y
                    #     boundaries_max_deepcopy[predicate.column_ids[0]] = max(max_y, boundaries_min_deepcopy[
                    #         predicate.column_ids[0]])
                    #
                    # if min_y > boundaries_min_deepcopy[predicate.column_ids[0]]:
                    #     # boundaries_min_deepcopy[predicate.column_ids[0]] = min_y
                    #     boundaries_min_deepcopy[predicate.column_ids[0]] = min(min_y, boundaries_max_deepcopy[
                    #         predicate.column_ids[0]])

            else:
                if min_x >= max_y:
                    start_index_next_bucket_per_condition[p_i] = j
                    break
                elif max_x <= min_y:
                    overlap[rect_j.estimation_range_id] = 0
                else:
                    overlap[rect_j.estimation_range_id] *= overlap_calculation(min_y, max_y, min_x, max_x, num_samples=num_samples)
                    """updating bounds"""
                    # if max_y < boundaries_max_deepcopy[predicate.column_ids[0]]:
                    #     # boundaries_max_deepcopy[predicate.column_ids[0]] = max_y
                    #     boundaries_max_deepcopy[predicate.column_ids[0]] = max(max_y, boundaries_min_deepcopy[
                    #         predicate.column_ids[0]])
                    #
                    # if min_y > boundaries_min_deepcopy[predicate.column_ids[0]]:
                    #     # boundaries_min_deepcopy[predicate.column_ids[0]] = min_y
                    #     boundaries_min_deepcopy[predicate.column_ids[0]] = min(min_y, boundaries_max_deepcopy[
                    #         predicate.column_ids[0]])
        """
            update the bounds with
        """
        # bucket1.min_boundaries = boundaries_min_deepcopy
        # bucket1.max_boundaries = boundaries_max_deepcopy


    total_card += math.ceil((overlap * buckets2_card).sum())
    if total_card > 0:
        return bucket1, total_card, None, start_index_next_bucket_per_condition, set_zeros

    return None, total_card, None, start_index_next_bucket_per_condition, set_zeros

def check_range_parallel_multijoin_fixed_sort_range(list_buckets_2, range_join_predicates, buckets2_card, num_samples, bucket1):

    total_card = 0

    overlap = np.ones(len(list_buckets_2[0])) * bucket1.num_points
    #boundaries_min_deepcopy = copy.deepcopy(bucket1.min_boundaries)
    #boundaries_max_deepcopy = copy.deepcopy(bucket1.max_boundaries)

    for p_i, predicate in enumerate(range_join_predicates):
        a = bucket1.min_boundaries[predicate.column_ids[0]]
        min_x = ne.evaluate(predicate.expressions[0])
        a = bucket1.max_boundaries[predicate.column_ids[0]]
        max_x = ne.evaluate(predicate.expressions[0])

        # take the list of buckets
        buckets2 = list_buckets_2[p_i]
        for j, rect_j in enumerate(buckets2):
            if overlap[rect_j.estimation_range_id] == 0:
                continue

            b = rect_j.min_boundaries[predicate.column_ids[1]]
            min_y = ne.evaluate(predicate.expressions[1])
            b = rect_j.max_boundaries[predicate.column_ids[1]]
            max_y = ne.evaluate(predicate.expressions[1])

            if predicate.operator == "<":
                if max_x <= min_y:
                    break
                elif min_x >= max_y:
                    overlap[rect_j.estimation_range_id] = 0
                else:
                    overlap[rect_j.estimation_range_id] *= overlap_calculation(min_x, max_x, min_y, max_y, num_samples=num_samples)
                    """updating bounds"""
                    #if max_y < boundaries_max_deepcopy[predicate.column_ids[0]]:
                    #    # # boundaries_max_deepcopy[predicate.column_ids[0]] = max_y
                    #    boundaries_max_deepcopy[predicate.column_ids[0]] = max(max_y, boundaries_min_deepcopy[predicate.column_ids[0]])
                    
                    #if min_y > boundaries_min_deepcopy[predicate.column_ids[0]]:
                    #    # # boundaries_min_deepcopy[predicate.column_ids[0]] = min_y
                    #    boundaries_min_deepcopy[predicate.column_ids[0]] = min(min_y, boundaries_max_deepcopy[predicate.column_ids[0]] )

            else:
                if min_x >= max_y:
                    break
                elif max_x <= min_y:
                    overlap[rect_j.estimation_range_id] = 0
                else:
                    overlap[rect_j.estimation_range_id] *= overlap_calculation(min_y, max_y, min_x, max_x, num_samples=num_samples)
                    """updating bounds"""
                    #if max_y < boundaries_max_deepcopy[predicate.column_ids[0]]:
                    #    # # boundaries_max_deepcopy[predicate.column_ids[0]] = max_y
                    #    boundaries_max_deepcopy[predicate.column_ids[0]] = max(max_y, boundaries_min_deepcopy[predicate.column_ids[0]])
                    
                    #if min_y > boundaries_min_deepcopy[predicate.column_ids[0]]:
                    #    # # boundaries_min_deepcopy[predicate.column_ids[0]] = min_y
                    #    boundaries_min_deepcopy[predicate.column_ids[0]] = min(min_y, boundaries_max_deepcopy[predicate.column_ids[0]] )
    """
        update the bounds with
    """
    # bucket1.min_boundaries = boundaries_min_deepcopy
    # bucket1.max_boundaries = boundaries_max_deepcopy

    # the estimated cardinality
    total_card += math.ceil((overlap * buckets2_card).sum())
    if total_card > 0:
        return bucket1, total_card, None

    return None, total_card, None

def overlap_calculation(x_min, x_max, y_min, y_max, type = 2, x_cell_obj=None, y_cell_obj=None, num_samples = 1):
    if type == 2:
        return overlap_percentage_estimate_proportion(x_min, x_max, y_min, y_max, num_samples=num_samples, x_cell_obj=x_cell_obj, y_cell_obj=y_cell_obj)
    else:
        print("Not implemented")
        exit(1)

def overlap_percentage_estimate_proportion(x_min, x_max, y_min, y_max, num_samples=1000, x_cell_obj=None, y_cell_obj=None):
    # Randomly sample points from each range
    if x_cell_obj is None:
        x_samples = np.random.uniform(x_min, x_max, num_samples)
    else:
        x_samples = x_cell_obj.sample_data(num_samples)
    if y_cell_obj is None:
        y_samples = np.random.uniform(y_min, y_max, num_samples)
    else:
        y_samples = y_cell_obj.sample_data(num_samples)
    # Count the number of sampled points where x is smaller than y
    count_smaller = np.sum(x_samples < y_samples)
    # Estimate the proportion
    return count_smaller / num_samples
    return count_smaller

def parallel_execution_multijoin_already_sorted_input(max_workers, lbs, rbs, rps, buckets2_card):
    res_buckets = list()
    tmp_results = dict()

    with Pool(processes=max_workers) as pool:
        func = partial(check_range_parallel_multijoin_fixed_sort_range, rbs, rps, buckets2_card, 100)

        results = pool.map(func, lbs)

    for return_cell, cell_estimate, _ in results:
        if return_cell is not None:
            # need to update the number of points for the cell
            # update the number of points to be multiplied with the cells of the next table
            # we compute the cardinality estimate by multiplying the number of points for every cell with the overlap between the cells
            # so always carry on the number of cells
            return_cell.num_points = cell_estimate
            res_buckets.append(return_cell)

            tmp_results[return_cell.id] = math.ceil(cell_estimate)
    return res_buckets, tmp_results

