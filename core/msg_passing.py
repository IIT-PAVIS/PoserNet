"""
Name: msg_passing.py
Description: The message-passing class, which defines how the graph is updated.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import torch

from core.my_mlp import MyMlp, MyNodeMlp
from utils import pose_algebra
from torch_scatter import scatter_mean


#####################################
# The custom message-passing class. #
#####################################
class MyMsgPasser(torch.nn.Module):
    """Custom message-passing class."""
    def __init__(self, perform_node_updates):
        super(MyMsgPasser, self).__init__()
        self.edge_mlp = MyMlp()
        self.node_mlp = MyNodeMlp()
        self.perform_node_updates = perform_node_updates

    def forward(self,
                x,
                edge_index,
                edge_index_NODES,
                edge_attr,
                det_features,
                # Indices
                temp_indices_NODES,
                temp_indices_EDGES,
                first_det_indices_NODES,
                first_det_indices_EDGES,
                second_det_indices_NODES,
                second_det_indices_EDGES,
                indices_for_aggregating_nodes_updates,
                last_pass):

        # 1. Perform the update on the edges features.
        updated_edge_attr = self.update_edges(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                              det_features=det_features,
                                              temp_indices_EDGES=temp_indices_EDGES,
                                              first_det_indices_EDGES=first_det_indices_EDGES,
                                              second_det_indices_EDGES=second_det_indices_EDGES)

        # 2. Perform the update on the node features.
        # Do not update the nodes if that is disabled or if this is the last round of message passing: the
        # output consists in the edge features, this part is not needed.
        if self.perform_node_updates and not last_pass:
            updated_x = self.message(edge_index=edge_index_NODES,
                                     x=x,
                                     edge_attr=updated_edge_attr,
                                     det_features=det_features,
                                     temp_indices_NODES=temp_indices_NODES,
                                     first_det_indices_NODES=first_det_indices_NODES,
                                     second_det_indices_NODES=second_det_indices_NODES,
                                     indices_for_aggregating_nodes_updates=indices_for_aggregating_nodes_updates)
        else:
            updated_x = x

        return updated_x, updated_edge_attr


    def update_edges(self,
                     x,
                     edge_index,
                     edge_attr,
                     det_features,
                     temp_indices_EDGES,
                     first_det_indices_EDGES,
                     second_det_indices_EDGES):
        """Update edge representations based on old edge features, source and destination node features, source and destination detection features.
        # Perform one update step for each object that is visible in both nodes connected by the edge, then average
        # the results.
        #
        # 1. Build the data structure that is given as input to the MLP:
        # [feats(node left), edge feats, feats(node right), feats(detection left), feats(detection right)]
        #
        # 2. Process the data with the MLP.
        #
        # 3. Average contributions for each edge = compute averages of sets of rows of the output of the MLP.        
        """

        # 1. Build the input data structure for the MLP.
        # This builds [node feats left, edge feats, node feats right] for all the graphs at the same time.
        # (edge_attr information is simply concatenated when batch_size>1, while the indices of the second graph in edge_index are moved forward by the number of edges of the first graph, and so on.)
        # It needs to be replicated for each detection present on both sides.
        node_id_left, node_id_right = edge_index
        temp = torch.cat([x[node_id_left], edge_attr, x[node_id_right]], dim=-1)

        data_for_mlp = torch.cat([temp[temp_indices_EDGES.squeeze(), :],
                                  det_features[first_det_indices_EDGES.squeeze(), :],
                                  det_features[second_det_indices_EDGES.squeeze(), :]], dim=-1)

        # 2. Run the MLP.
        single_updates = self.edge_mlp(data_for_mlp)

        # 3. Average the contributions for each edge.
        updated_edge_attr = scatter_mean(single_updates, dim=0, index=temp_indices_EDGES)

        return updated_edge_attr


    def message(self, x, edge_index, edge_attr, det_features, temp_indices_NODES,
                first_det_indices_NODES, second_det_indices_NODES, indices_for_aggregating_nodes_updates):
        """Custom message function, which overrides the default one.
        """
        # x: node features before the update.
        # edge_attr: edge features (before and after the update, they are not changed here)

        node_id_left, node_id_right = edge_index
        x_i = x[node_id_left]
        x_j = x[node_id_right]
        # x_i: node features corresponding to the source node of each edge.
        # x_j: node features corresponding to the destination node of each edge.

        # Concatenate x_j, x_i and the edge features: they are all [n_edges x something],
        # then pass each row through an MLP.

        # Double the size of edge features, fill the second part with the inverse of the transformation encoded in the
        # corresponding slot of the first part. This has to be done so that node updates are directional:
        # When updating N1, with the information coming from N0, the transformation has to be used as it is encoded in
        # edge_attr. (This happens when index source < index dest).
        # When updating N0 with the information coming from N1, the transformation has to be inverted.
        replicated_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # Vectorised edge replication, with pose inversion (The last element is not used at the moment, but it is copied).
        replicated_edge_attr[edge_attr.shape[0]:, 0:7] = pose_algebra.invert_pose_quaternion_vectorised(edge_attr[0:edge_attr.shape[0], 0:7])
        replicated_edge_attr[edge_attr.shape[0]:, 7] = edge_attr[0:edge_attr.shape[0], 7]

        # Build the temporary structure with [image features, edge features, image features], which will be replicated later.
        # This structure has one row per edge, per direction.
        temp = torch.cat([x_i, replicated_edge_attr, x_j], dim=-1)

        # Add the information on matched detections to each row (rows of temp are replicated as needed).
        data_for_mlp = torch.cat([temp[temp_indices_NODES.squeeze(), :], det_features[first_det_indices_NODES.squeeze(), :], det_features[second_det_indices_NODES.squeeze(), :]], dim=-1)

        # 2. Run the MLP
        single_updates = self.node_mlp(data_for_mlp)

        # 3. Compute the average of the contributions for each edge of the current graph and store the
        #    information in the correct part of the output tensor.
        updated_node_attr = scatter_mean(single_updates, dim=0, index=indices_for_aggregating_nodes_updates)

        return updated_node_attr
