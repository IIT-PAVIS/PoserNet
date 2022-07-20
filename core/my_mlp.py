"""
Name: my_mlp.py
Description: The Multi-Layer Perceptrons used for processing data during the edge update and the node update.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
from torch import nn
from core.embeddings_definition import node_features_size, det_features_size, edge_features_size


# The MLP which updates each edge.
class MyMlp(nn.Module):
    def __init__(self):
        super(MyMlp, self).__init__()

        n_input = (node_features_size + det_features_size) * 2 + edge_features_size
        n_output = edge_features_size
        layers = []
        layers.append(nn.Linear(n_input, 32))
        nn.init.xavier_uniform_(layers[0].weight)
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Linear(32, 32))
        nn.init.xavier_uniform_(layers[2].weight)
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Linear(32, n_output))
        nn.init.xavier_uniform_(layers[4].weight)

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)


# The MLP which updates each node.
class MyNodeMlp(nn.Module):
    def __init__(self):
        super(MyNodeMlp, self).__init__()

        n_input = (node_features_size + det_features_size) * 2 + edge_features_size
        n_output = node_features_size
        layers = []
        layers.append(nn.Linear(n_input, 32))
        nn.init.xavier_uniform_(layers[0].weight)
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Linear(32, 32))
        nn.init.xavier_uniform_(layers[2].weight)
        layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Linear(32, n_output))
        nn.init.xavier_uniform_(layers[4].weight)

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)
