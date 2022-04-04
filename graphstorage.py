"""
GraphStorage Class in DGL 
"""
import cugraph
import dgl
from cugraph.experimental import PropertyGraph
import pandas as pd
import numpy as np
import cudf
import cupy


class GraphStorage(object):
    # the input g here is the cugraph graphs? whats ndata and edata?
    def __init__(self, g, ndata = None, edata = None):
        # g is the cugraph property graph
        self.graphstore =  cugraph.gnn.CuGraphStore(graph = g)
        self._edata = g.edata
        self._ndata = g.ndata

    @property
    def ndata(self):
        return self._ndata

    @property
    def edata(self):
        return self._edata

    def get_node_storage(self, key, ntype=None):
        # ndata is dataframe, ntype = merchants, key = merchant_id
        #(ndata[ndata['_TYPE_']== 'merchants']['merchant_id'])
        return self._ndata [self._ndata['_TYPE_'] == ntype][key]
        #select_vertices not correct function

    def get_edge_storage(self, key, etype=None):
        # edata is dataframe
        return self._edata [self._edata['_TYPE_'] == etype][key]
        #select_edges

    # Required for checking whether a single dict is allowed for ndata and edata.
    #@property
    #def ntypes(self):
    #    pass
    # I think we always allow that?

    @property
    #def canonical_etypes(self):
    #    pass

    #def etypes(self):
        #return [etype[1] for etype in self.canonical_etypes]

    # from cugraph to DGL using 
    def toDGL(graph):
        # input is cugraph graph
        # output is DGL graph
        edgelist = graph.edges()
        src = cupy.asarray(edgelist['src'])
        dst = cupy.asarray(edgelist['dst'])
        g_dgl = dgl.graph((src, dst))
        return g_dgl


    def sample_neighbors(self, seed_nodes, fanout, edge_dir='in', prob=None,
                         exclude_edges=None, replace=False, output_device=None):
        """Return a DGLGraph which is a subgraph induced by sampling neighboring edges of
        the given nodes.
        See ``dgl.sampling.sample_neighbors`` for detailed semantics.
        Parameters
        ----------
        seed_nodes : Tensor or dict[str, Tensor]
            Node IDs to sample neighbors from.
            This argument can take a single ID tensor or a dictionary of node types and ID tensors.
            If a single tensor is given, the graph must only have one type of nodes.
        fanout : int or dict[etype, int]
            The number of edges to be sampled for each node on each edge type.
            This argument can take a single int or a dictionary of edge types and ints.
            If a single int is given, DGL will sample this number of edges for each node for
            every edge type.
            If -1 is given for a single edge type, all the neighboring edges with that edge
            type will be selected.
        prob : str, optional
            Feature name used as the (unnormalized) probabilities associated with each
            neighboring edge of a node.  The feature must have only one element for each
            edge.
            The features must be non-negative floats, and the sum of the features of
            inbound/outbound edges for every node must be positive (though they don't have
            to sum up to one).  Otherwise, the result will be undefined.
            If :attr:`prob` is not None, GPU sampling is not supported.
        exclude_edges: tensor or dict
            Edge IDs to exclude during sampling neighbors for the seed nodes.
            This argument can take a single ID tensor or a dictionary of edge types and ID tensors.
            If a single tensor is given, the graph must only have one type of nodes.
        replace : bool, optional
            If True, sample with replacement.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            A sampled subgraph with the same nodes as the original graph, but only the sampled neighboring
            edges.  The induced edge IDs will be in ``edata[dgl.EID]``.
        """
        #sampled_graph = cugraph_sampler.cugraphSampler(XXXXX, )
        # return type is cupy array
        parents_nodes, children_nodes = self.graphstore.sample_neighbors(seed_nodes, fanout, edge_dir='in', prob=None, replace=False)
        # construct dgl graph, want to double check if children and parents are in the correct order
        sampled_graph = dgl.graph ((children_nodes,parents_nodes))
        # to device function move the dgl graph to desired devices
        sampled_graph.to_device (output_device)
        return sampled_graph

    # Required in Cluster-GCN
    
    def subgraph(self, nodes, relabel_nodes=False, output_device=None):
        """Return a subgraph induced on given nodes.
        This has the same semantics as ``dgl.node_subgraph``.
        Parameters
        ----------
        nodes : nodes or dict[str, nodes]
            The nodes to form the subgraph. The allowed nodes formats are:
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag indicating whether
              node :math:`i` is in the subgraph.
            If the graph is homogeneous, one can directly pass the above formats.
            Otherwise, the argument must be a dictionary with keys being node types
            and values being the node IDs in the above formats.
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the specified node set
            and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        sampled_cugraph = self.graphstore.node_subgraph(nodes)
        # the return type is cugraph subgraph
        sample_graph = toDGL(sampled_cugraph)
        sampled_graph.to_device (output_device)
        return sampled_graph


    # Required in Link Prediction
    # we don't need to implement this function as advised by DGL
    #def edge_subgraph(self, edges, relabel_nodes=False, output_device=None):
        """Return a subgraph induced on given edges.
        This has the same semantics as ``dgl.edge_subgraph``.
        Parameters
        ----------
        edges : edges or dict[(str, str, str), edges]
            The edges to form the subgraph. The allowed edges formats are:
            * Int Tensor: Each element is an edge ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag indicating whether
              edge :math:`i` is in the subgraph.
            If the graph is homogeneous, one can directly pass the above formats.
            Otherwise, the argument must be a dictionary with keys being edge types
            and values being the edge IDs in the above formats.
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the specified node set
            and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
         


# no need to implement the 3 functions below
    # Required in Link Prediction negative sampler
    def find_edges(self, edges, etype=None, output_device=None):
        """Return the source and destination node IDs given the edge IDs within the given edge type.
        """
        # edges are a range of edge IDs, for example 0-100
        selected_edges = self._edata[self._edata['_TYPE_'] == etype].iloc[edges]
        src_nodes = selected_edges['_SRC_']
        dst_nodes = selected_edges['_DST_']

        return src_nodes, dst_nodes


    # Required in Link Prediction negative sampler
    def num_nodes(self, ntype):
        """Return the number of nodes for the given node type."""
        # use graphstore function
        return self._ndata[self._ndata['_TYPE_']== ntype].shape[0]
        

    #def global_uniform_negative_sampling(self, num_samples, exclude_self_loops=True,
                                         replace=False, etype=None):
        """Per source negative sampling as in ``dgl.dataloading.GlobalUniform``"""

