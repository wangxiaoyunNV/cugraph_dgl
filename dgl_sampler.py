# example of dgl
import dgl
g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))
sg = dgl.sampling.sample_neighbors(g, [0, 1], 1)
print(sg.edges())
print(sg.edata)
print(sg.edata[dgl.EID])
block0 =dgl.to_block(sg,[0,1])
print (block0)
