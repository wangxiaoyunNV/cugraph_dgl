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

pg.add_edge_data(cora_M, vertex_col_names=("0","1"))
pg.add_vertex_data(cora_content1, vertex_col_name = "0")

print(pg._vertex_prop_dataframe)
print(pg._edge_prop_dataframe)

# create graph storage
gstore = cugraph.gnn.CuGraphStore(graph=pg)




