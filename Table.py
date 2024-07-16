import numpy as np

OPS = {
    ">": (lambda x, y: x > y),
    "<": (lambda x, y: x < y),
    "<=": (lambda x, y: x <= y),
    ">=": (lambda x, y: x >= y),
    "=": (lambda x, y: x == y)
}

'''
    Class representing a table containing rows and header
'''
class Table:
    def __init__(self, table_name, header, rows, column_type, index_enabled = False):
        self.table_name = table_name
        self.header = header
        # each row is an array of integers, and rows is an array of
        # rows, all of the same length
        self.rows = np.array(rows, dtype='O')
        self.nb_rows = len(self.rows)
        # print(self.rows)
        self.column_type = column_type
        # we'll be inefficient and build an index for every column
        self.indices = {}
        self.distinct_vals = {}
        print('The header of the table is: ')
        print(self.header)

        self.size = len(self.rows)
        self.index_enabled = index_enabled
        if index_enabled:
            print("Implement creating an index")
            self.new_rows = None

    def create_index(self, data):
        '''
        Creates an index over the columns in the table. For each value it needs the tuple ids where it appears.
        :param data: the full data
        :return: hashmap index / inverted index
        '''
        hashmap = {}
        for i, d in enumerate(data):
            if d in hashmap:
                hashmap[d].append(i)
            else:
                hashmap[d] = [i]
        return hashmap

    def get_col_id(self, col_name):
        '''
        Returns an index of a column name in the header
        :param col_name: the column name of the id we are searching for
        :return: the index of the column name
        '''
        for id in range(len(self.header)):
            if self.header[id] == col_name:
                return id

def convert_row(row, table_i):
    new_row = []
    for r_id, r_i in enumerate(row):
        # If the data is numerical we do not need anything
        if table_i.header[r_id] not in ['account'] and (isinstance(r_i, int) or isinstance(r_i, float)):
            new_row.append(r_i)
            continue

        if table_i.header[r_id] in table_i.column_mapper:
            column_mapper_i = table_i.column_mapper[table_i.header[r_id]]
            new_row.append(column_mapper_i[r_i])
            continue

    return new_row