"""
Name: my_graph.py
Description: Code adapted from PyTorch Geometric (PyG). We recommend that you use that package directly, for your projects.
             PyG is distributed with an MIT license.
             Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>.
             That license is of the same type as that used for PoserNet, please refer to the LICENSE file for details,
             or look at the LICENSE file for PyG on its GitHub page: https://github.com/pyg-team/pytorch_geometric.
-----
Authors: Matthias Fey, Jiaxuan You, Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import re
import torch

########################################################################################################################
# Description of the graphs                                                                                            #
########################################################################################################################
# One node represents one image = one camera pose.
# The size of the embeddings is defined in embeddings_definition.py.
#
# Data structures:
#   x = the node embeddings for the current graph [n_nodes x nodes_embeddings_size].
#   x_det_ptr = pointers that go from a node (x) to the detections (det_features) associated with that node [n_nodes x max_detections_per_node].
#   det_features = one row for each detection for the current graph, encode BB geometry. [n_dets x det_features_size]
#
#   edge_index = connectivity information, the top row indicates the source node, the bottom one the destination.
#                [2 x n_edges] when created, but later it is duplicated with edges going in the opposite direction.
#                [2 x 2*n_edges].
#   edge_attr = the edge embeddings. There is only one embedding for one edge, in spite of it appearing twice in edge_index.
#   y = the GT values for what we want to estimate: edge_attr.


class MyGraph:
    def __init__(self,
                 x=None,
                 y=None,
                 edge_index=None, edge_index_NODES=None, edge_attr=None, det_features=None, x_det_ptr=None, n_dets=None,
                 temp_indices_NODES=None, temp_indices_EDGES=None,
                 first_det_indices_NODES=None, first_det_indices_EDGES=None,
                 second_det_indices_NODES=None, second_det_indices_EDGES=None,
                 indices_for_aggregating_nodes_updates=None,
                 n_edges=None,
                 n_nodes=None,
                 graph_id=None,
                 gt_absolute_poses=None):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_index_NODES = edge_index_NODES
        self.edge_attr = edge_attr
        self.det_features = det_features
        self.x_det_ptr = x_det_ptr
        self.n_dets = n_dets
        self.temp_indices_NODES = temp_indices_NODES
        self.temp_indices_EDGES = temp_indices_EDGES
        self.first_det_indices_NODES = first_det_indices_NODES
        self.first_det_indices_EDGES = first_det_indices_EDGES
        self.second_det_indices_NODES = second_det_indices_NODES
        self.second_det_indices_EDGES = second_det_indices_EDGES
        self.indices_for_aggregating_nodes_updates = indices_for_aggregating_nodes_updates
        self.n_edges = n_edges
        self.n_nodes = n_nodes
        self.graph_id = graph_id
        self.gt_absolute_poses = gt_absolute_poses

        if not x is None:
            self.__num_nodes__ = len(x)
            self.num_nodes = len(x)
        else:
            self.__num_nodes__ = 0
            self.num_nodes = 0


    def cuda(self):
        self.x = self.x.cuda()
        self.y = self.y.cuda()
        self.edge_index = self.edge_index.cuda()
        self.edge_index_NODES = self.edge_index_NODES.cuda()
        self.edge_attr = self.edge_attr.cuda()
        self.det_features = self.det_features.cuda()
        self.x_det_ptr = self.x_det_ptr.cuda()
        self.n_dets = self.n_dets.cuda()
        self.temp_indices_NODES = self.temp_indices_NODES.cuda()
        self.temp_indices_EDGES = self.temp_indices_EDGES.cuda()
        self.first_det_indices_NODES = self.first_det_indices_NODES.cuda()
        self.first_det_indices_EDGES = self.first_det_indices_EDGES.cuda()
        self.second_det_indices_NODES = self.second_det_indices_NODES.cuda()
        self.second_det_indices_EDGES = self.second_det_indices_EDGES.cuda()
        self.indices_for_aggregating_nodes_updates = self.indices_for_aggregating_nodes_updates()
        self.n_edges = self.n_edges.cuda()
        self.n_nodes = self.n_nodes.cuda()
        self.graph_id = self.graph_id
        self.gt_absolute_poses = self.gt_absolute_poses.cuda()



    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        return data

    def to_dict(self):
        return {key: item for key, item in self}

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Concatenate `*index*` and `*face*` attributes in the last dimension.
        if bool(re.search('(index|face)', key)):
            return -1
        # # By default, concatenate sparse matrices diagonally.
        # elif isinstance(value, SparseTensor):
        #     return (0, 1)
        return 0

    def __inc__(self, key, value):
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        return self.num_nodes if bool(re.search('(index|face)', key)) else 0

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.

        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'normal', 'batch'):
            # if isinstance(item, SparseTensor):
            #     return item.size(0)
            # else:
            return item.size(self.__cat_dim__(key, item))
        if hasattr(self, 'adj'):
            return self.adj.size(0)
        if hasattr(self, 'adj_t'):
            return self.adj_t.size(1)
        # if self.face is not None:
        #     logging.warning(__num_nodes_warn_msg__.format('face'))
        #     return maybe_num_nodes(self.face)
        # if self.edge_index is not None:
        #     logging.warning(__num_nodes_warn_msg__.format('edge'))
        #     return maybe_num_nodes(self.edge_index)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        """
        Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges.
        """
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.__cat_dim__(key, item))
        for key, item in self('adj', 'adj_t'):
            return item.nnz()
        return None

    @property
    def num_faces(self):
        r"""Returns the number of faces in the mesh."""
        if self.face is not None:
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the graph."""
        if self.edge_attr is None:
            return 0
        return 1 if self.edge_attr.dim() == 1 else self.edge_attr.size(1)

    def is_directed(self):
        r"""Returns :obj:`True`, if graph edges are directed."""
        return not self.is_undirected()

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def cpu(self, *keys):
        r"""Copies all attributes :obj:`*keys` to CPU memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.cpu(), *keys)

    def cuda(self, device=None, non_blocking=False, *keys):
        r"""Copies all attributes :obj:`*keys` to CUDA memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(
            lambda x: x.cuda(device=device, non_blocking=non_blocking), *keys)

