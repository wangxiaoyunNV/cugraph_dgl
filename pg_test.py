import time
import gc

import pytest
import pandas as pd
import numpy as np
import cudf
from cudf.testing import assert_frame_equal, assert_series_equal

import cugraph
from cugraph.generators import rmat
from cugraph.tests import utils
import dgl
import cupy

dataset1 = {
    "merchants": [
        ["merchant_id", "merchant_location", "merchant_size", "merchant_sales",
         "merchant_num_employees", "merchant_name"],
        [(11, 78750, 44, 123.2, 12, "north"),
         (4, 78757, 112, 234.99, 18, "south"),
         (21, 44145, 83, 992.1, 27, "east"),
         (16, 47906, 92, 32.43, 5, "west"),
         (86, 47906, 192, 2.43, 51, "west"),
         ]
     ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [(89021, 78757, 0),
         (32431, 78750, 1),
         (89216, 78757, 1),
         (78634, 47906, 0),
         ]
     ],
    "taxpayers": [
        ["payer_id", "amount"],
        [(11, 1123.98),
         (4, 3243.7),
         (21, 8932.3),
         (16, 3241.77),
         (86, 789.2),
         (89021, 23.98),
         (78634, 41.77),
         ]
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num", "card_type"],
        [(89021, 11, 33.2, 1639084966.5513437, 123456, "MC"),
         (89216, 4, None, 1639085163.481217, 8832, "CASH"),
         (78634, 16, 72.0, 1639084912.567394, 4321, "DEBIT"),
         (32431, 4, 103.2, 1639084721.354346, 98124, "V"),
         ]
     ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [(89216, 89021, 9),
         (89216, 32431, 9),
         (32431, 78634, 8),
         (78634, 89216, 8),
         ]
     ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [(89216, 78634, 11, 5),
         (89021, 89216, 4, 4),
         (89021, 89216, 21, 3),
         (89021, 89216, 11, 3),
         (89021, 78634, 21, 4),
         (78634, 32431, 11, 4),
         ]
     ],
}


def dataset1_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = request
    from cugraph.experimental import PropertyGraph

    (merchants, users, taxpayers,
     transactions, relationships, referrals) = dataset1.values()

    pG = PropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    pG.add_vertex_data(dataframe_type(columns=merchants[0],
                                      data=merchants[1]),
                       type_name="merchants",
                       vertex_col_name="merchant_id",
                       property_columns=None)
    pG.add_vertex_data(dataframe_type(columns=users[0],
                                      data=users[1]),
                       type_name="users",
                       vertex_col_name="user_id",
                       property_columns=None)
    pG.add_vertex_data(dataframe_type(columns=taxpayers[0],
                                      data=taxpayers[1]),
                       type_name="taxpayers",
                       vertex_col_name="payer_id",
                       property_columns=None)

    pG.add_edge_data(dataframe_type(columns=transactions[0],
                                    data=transactions[1]),
                     type_name="transactions",
                     vertex_col_names=("user_id", "merchant_id"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=relationships[0],
                                    data=relationships[1]),
                     type_name="relationships",
                     vertex_col_names=("user_id_1", "user_id_2"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=referrals[0],
                                    data=referrals[1]),
                     type_name="referrals",
                     vertex_col_names=("user_id_1",
                                       "user_id_2"),
                     property_columns=None)

    return pG


def toDGL(graph):
    # input is cugraph graph
    # output is DGL graph
    edgelist = graph.edges()
    src = cupy.asarray(edgelist['src'])
    dst = cupy.asarray(edgelist['dst'])
    g_dgl = dgl.graph((src, dst))
    return g_dgl



if __name__ == '__main__':
    pG= dataset1_PropertyGraph(cudf.DataFrame)
    print ("edges prop")
    print (pG._edge_prop_dataframe)
    print ("node prop")
    print (pG._vertex_prop_dataframe)
    ndata = pG._vertex_prop_dataframe
    print (ndata['_TYPE_']== 'merchants')
    print (ndata[ndata['_TYPE_']== 'merchants'])
    print (ndata[ndata['_TYPE_']== 'merchants']['merchant_id'])

    edata = pG._edge_prop_dataframe
    selected_edges = edata[edata['_TYPE_'] == 'transactions'].iloc[[1,2]]
    print(selected_edges['_SRC_'])
    
    print(ndata[ndata['_TYPE_']== 'merchants'].shape[0])

    g = cugraph.Graph()
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    g.from_cudf_edgelist(df, source="a", destination="b")
    print (g.nodes())
    print (g.edges())
    edgelist = g.edges()
    print (edgelist)
    #g_dgl = dgl.DGLGraph(edgelist)
    print("dst", edgelist['dst'])
    src = cupy.asarray(edgelist['src'])
    dst = cupy.asarray(edgelist['dst'])
    print (dst)
    g_dgl = dgl.graph((src, dst))
    print (g_dgl)

