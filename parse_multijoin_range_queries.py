import numpy as np
import csv

def parse_query(line, tables_order):
    # take all the parts
    line_split = line.strip().split("#")
    # tables are 0
    tables = line_split[0].strip()
    # join pairs are 1
    join_part = line_split[1].strip()
    # predicates per table are 2
    query_predicates = line_split[2].strip()
    # cardinality
    query_cardinality = int(line_split[3].strip())

    # extracting info about the tables
    query_tables, table_alias_to_table_map = get_tables_to_alias_mapping(tables.split(','))
    # extracting info about the join conditions
    join_conditions_map = get_join_conditions(join_part.split(','), tables_order, table_alias_to_table_map)
    # extracting info about the query predicates
    query_predicates_per_table = get_predicates_per_table(query_predicates)

    # create the query object with everythig
    query_object = MultiJoinQuery(query_tables, table_alias_to_table_map, join_conditions_map,
                                  query_predicates_per_table, query_cardinality)
    return query_object

def parse_queries(path_to_file, tables_order):
    all_queries = list()
    with open(path_to_file, 'r') as query_file:
        for line_id, line in enumerate(query_file):
            # take all the parts
            line_split = line.strip().split("#")
            # tables are 0
            tables = line_split[0].strip()
            # join pairs are 1
            join_part = line_split[1].strip()
            # predicates per table are 2
            query_predicates = line_split[2].strip()
            # cardinality
            query_cardinality = int(line_split[3].strip())

            # extracting info about the tables
            query_tables, table_alias_to_table_map = get_tables_to_alias_mapping(tables.split(','))
            # extracting info about the join conditions
            join_conditions_map = get_join_conditions(join_part.split(','), tables_order, table_alias_to_table_map)
            # extracting info about the query predicates
            query_predicates_per_table = get_predicates_per_table(query_predicates)

            # create the query object with everythig
            query_object = MultiJoinQuery(query_tables, table_alias_to_table_map, join_conditions_map, query_predicates_per_table, query_cardinality)

            all_queries.append(query_object)

    return all_queries

def get_predicates_per_table(query_predicates):
    query_predicates_split = ['{}'.format(x) for x in list(csv.reader([query_predicates], delimiter=',', quotechar="'"))[0]]

    total_size = len(query_predicates_split)
    pred_iter = 0

    query_predicates_per_table = dict()
    while pred_iter < total_size - 1:

        # always in pairs of 3
        pred_column = query_predicates_split[pred_iter]
        pred_iter += 1
        pred_sign = query_predicates_split[pred_iter]
        pred_iter += 1
        pred_val = query_predicates_split[pred_iter]
        # next query predicate start
        pred_iter += 1

        table_alias = pred_column.split('.')[0].strip()

        all_preds_for_table = list()
        if table_alias in query_predicates_per_table.keys():
            all_preds_for_table = query_predicates_per_table[table_alias]

        all_preds_for_table.append((pred_column, pred_sign, pred_val))

        query_predicates_per_table[table_alias] = all_preds_for_table

    return query_predicates_per_table

def get_tables_to_alias_mapping(tables):
    query_tables = set()
    table_alias_to_table_map = dict()
    for table_info in tables:
        table_info_split = table_info.split(' ')
        table_name = table_info_split[0].strip()
        table_alias = table_info_split[1].strip()

        query_tables.add(table_name)
        table_alias_to_table_map[table_alias] = table_name

    return query_tables, table_alias_to_table_map

def get_join_conditions(join_conditions, tables_order, tables_map):
    join_conditions_map = dict()
    for join_condition in join_conditions:
        # print(join_condition)
        join_sign = '>'
        if '>=' in join_condition:
            condition_split = join_condition.strip().split('>=')
            join_sign = '>='
        elif '<=' in join_condition:
            condition_split = join_condition.strip().split('<=')
            join_sign = '<='
        elif '=' in join_condition:
            condition_split = join_condition.strip().split('=')
            join_sign = '='
        elif '<' in join_condition:
            condition_split = join_condition.strip().split('<')
            join_sign = '<'
        elif '>' in join_condition:
            condition_split = join_condition.strip().split('>')
        else:
            raise Exception('Join sign not supported, has to be one of [>=, <=, =, >, <]')

        # the left and right side of the join condition
        left_side = condition_split[0].strip()
        right_side = condition_split[1].strip()

        # the left and right table name
        left_table_name = left_side.split('.')[0].strip()
        right_table_name = right_side.split('.')[0].strip()
        if 'sin(' in left_table_name:
            left_table_name = left_table_name.replace('sin(', '')

        if 'sin(' in right_table_name:
            right_table_name = right_table_name.replace('sin(', '')

        # the two tables involved in the query sorted by some order so that even if they appear in different order
        # it will be the same pair for different queries
        tables_pair = ""
        if tables_order[tables_map[left_table_name]] < tables_order[tables_map[right_table_name]]:
            tables_pair = left_table_name + "," + right_table_name
        else:
            tables_pair = right_table_name + "," + left_table_name

        # the class object with all info
        join_pair = JoinPair(left_table_name, right_table_name, join_sign, left_side, right_side)
        existing_predicates = list()
        if tables_pair in join_conditions_map.keys():
            existing_predicates = join_conditions_map[tables_pair]
        existing_predicates.append(join_pair)

        join_conditions_map[tables_pair] = existing_predicates

    """
        Check which tables are used in the successive join conditions 
    """
    all_table_pairs = list(join_conditions_map.keys())
    for i, tables_pair in enumerate(join_conditions_map.keys()):
        j = i + 1
        tables_pair_split = tables_pair.split(',')
        left_table, right_table = tables_pair_split[0].strip(), tables_pair_split[1].strip()
        while j < len(join_conditions_map.keys()):
            next_join_condition = all_table_pairs[j]
            if left_table in next_join_condition:
                for join_cond_obj in join_conditions_map[tables_pair]:
                    join_cond_obj.table_used_in_succcessive_joins[left_table] = list()
            elif right_table in next_join_condition:
                for join_cond_obj in join_conditions_map[tables_pair]:
                    join_cond_obj.table_used_in_succcessive_joins[right_table] = list()
            j += 1


    return join_conditions_map



OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '!=': np.not_equal
}

class JoinPair:
    def __init__(self, left_table, right_table, sign, left_side, right_side, ):
        self.left_table = left_table
        self.right_table = right_table
        self.sign = sign
        self.left_side = left_side
        self.right_side = right_side
        self.table_used_in_succcessive_joins = dict()

    def __str__(self) -> str:
        return f"{{ LT:{self.left_table}, RT:{self.right_table}, Sign:{self.sign}, LS:{self.left_side}, RS:{self.right_side}, tables used in next joins:{self.table_used_in_succcessive_joins}  }}"

class MultiJoinQuery:
    def __init__(self, tables, table_alias_to_table_map, join_pairs, query_predicates_per_table, query_cardinality):
        self.tables = tables
        self.table_alias_to_table_map = table_alias_to_table_map
        self.join_pairs = join_pairs
        self.query_predicates_per_table = query_predicates_per_table
        self.query_cardinality = query_cardinality
        self.grid_index_query_min = dict()
        self.grid_index_query_max = dict()
        self.ar_model_queries = dict()

    def print(self):
        print('-------------QUERY-------------')
        print(f"T:{self.tables}")
        print(f"T_Aliases:{self.table_alias_to_table_map}")
        for join_condition in self.join_pairs:
            print(join_condition)
            for tmp_c in self.join_pairs[join_condition]:
                print(f'\t {str(tmp_c)}')
        print(f"PredicatesTable:{self.query_predicates_per_table}")
        print(f"Cardinality:{self.query_cardinality}")

    def __str__(self) -> str:
        return f"{{ T:{self.tables}, T_Aliases:{self.table_alias_to_table_map}, PredicatesTable:{self.query_predicates_per_table}, Cardinality:{self.query_cardinality}  }}"
