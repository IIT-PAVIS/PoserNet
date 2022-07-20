"""
Name: pose_refiner.py
Description: The PoseRefiner module, which applies the message-passing scheme twice for updating the information on a graph.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import torch
from core.msg_passing import MyMsgPasser


###########################################
# The Pose Refiner PyTorch module, which  #
# uses the custom message-passing class   #
###########################################
class PoseRefiner0(torch.nn.Module):
    def __init__(self, perform_node_updates, type_of_input_estimates):
        super(PoseRefiner0, self).__init__()
        self.msg_passer = MyMsgPasser(perform_node_updates)
        self.type_of_input_estimates = type_of_input_estimates

    def forward(self, data):
        # Apply the message-passing twice (the second time update only the edges).
        x, edge_attr = self.msg_passer.forward(x=data.x,
                                               edge_index=data.edge_index,
                                               edge_index_NODES=data.edge_index_NODES,
                                               edge_attr=data.edge_attr,
                                               det_features=data.det_features,
                                               # Indices
                                               temp_indices_NODES=data.temp_indices_NODES,
                                               temp_indices_EDGES=data.temp_indices_EDGES,
                                               first_det_indices_NODES=data.first_det_indices_NODES,
                                               first_det_indices_EDGES=data.first_det_indices_EDGES,
                                               second_det_indices_NODES=data.second_det_indices_NODES,
                                               second_det_indices_EDGES=data.second_det_indices_EDGES,
                                               indices_for_aggregating_nodes_updates=data.indices_for_aggregating_nodes_updates,
                                               last_pass=False)

        x, edge_attr = self.msg_passer.forward(x=x,
                                               edge_index=data.edge_index,
                                               edge_index_NODES=data.edge_index_NODES,
                                               edge_attr=edge_attr,
                                               det_features=data.det_features,
                                               # Indices
                                               temp_indices_NODES=data.temp_indices_NODES,
                                               temp_indices_EDGES=data.temp_indices_EDGES,
                                               first_det_indices_NODES=data.first_det_indices_NODES,
                                               first_det_indices_EDGES=data.first_det_indices_EDGES,
                                               second_det_indices_NODES=data.second_det_indices_NODES,
                                               second_det_indices_EDGES=data.second_det_indices_EDGES,
                                               indices_for_aggregating_nodes_updates=data.indices_for_aggregating_nodes_updates,
                                               last_pass=True)

        return edge_attr  # These are the estimated edge representations.
