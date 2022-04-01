import cugraph
import cudf
from cugraph.experimental import PropertyGraph

cora_M = cudf.read_csv('/home/xiaoyunw/cugraph/datasets/cora/cora.cites', sep = '\t', header = None)
cora_content = cudf.read_csv('/home/xiaoyunw/cugraph/datasets/cora/cora.content', sep = '\t', header = None)
# the last column is true label
cora_content1 = cora_content.drop (columns = '1434')
# add weight into graph
cora_M['weight'] = 1.0

# add features to nodes and edges
pg = PropertyGraph()

pg.add_edge_data(cora_M, vertex_id_columns=("0","1"))
pg.add_vertex_data(cora_content1, vertex_id_column = '0')
# create graph storage
gstore = cugraph.gnn.CuGraphStore(graph=pg)

# select node features 
def get_node_storage(pg, key, ntype = None):
    # key is the nodes feature col name, ntype is node type.
    # this function returns a col of all nodes features.
    selection = pg.select_vertices("(_TYPE_==ntype) & (pg.vertex_col_names==key)")
    return selection

# select edge features
def get_edge_storage(pg, key, etype=None):
    # key is the nodes feature col name, etype is the edge type. 
    # this function returns a col of all edges features.
    tcn = pg.type_col_name
    selection = pg.select_edges("(_TYPE_=etype) & {tcn}==key")
    return selection


# 


# get samples of graph in graph store
sampled_graph = gstore.sample_neighbors(seeds, fanout = 5)

egonet_edgelist,seeds_offsets = gstore.egonet(seeds, 1)





