"""
    Deep Unsupervised Cardinality Estimation Source Code

    Source Code used as is or modified from the above mentioned source
"""

"""Dataset registrations."""
import common

def LoadTableSpecial(data, table_name='customer', do_compression=False, column_names='grid_cell,c_custkey,c_nationkey', folder_name=''):
    cols = column_names.split(',')
    type_casts = {}
    irrelevant_columns = []
    return common.CsvTable(table_name, '', cols=cols, do_compression=do_compression, num_submterms=2,
                           comp_threshold=2000,
                           type_casts=type_casts, read_intel=True, separator='|', folder_name=folder_name,
                           partition_id=-1, read_all=True, irrelevant_columns=irrelevant_columns,
                           reindex_cols=[], custom_data=data)

