"""
Name: embeddings_definition.py
Description: Lists the size of the embeddings for the components of PoserNet: nodes, edges and detections.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""

node_features_size = 4  # Image width, height, focal length and first radial distortion parameter, somehow normalised.
det_features_size  = 4  # Normalised bounding box (horizontal centre, vertical centre, width, height).
edge_features_size = 8  # Relative orientation quaternion [4], direction of translation [3], not set [1].

