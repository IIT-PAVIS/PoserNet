"""
Name: pose_loss.py
Description: Defines the way the loss is computed, by computed a weighted sum of four components.
-----
Authors: Matteo Taiana, Matteo Toso.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
# Path hack used to import scripts from sibling directories.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import math
from utils import pose_algebra


tPi = torch.tensor(math.pi).to(device='cuda')


class PoseLoss(torch.nn.Module):

    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, x, y):

        # 1. Compute quaternion norms, quaternion normalisation loss.
        quat_norms = torch.norm(x[:, 0:4], dim=1)
        quaternion_normalisation_loss = torch.mean(torch.abs(quat_norms - 1))  # Norm of a unit quaternion is 1. Anything different is a mistake.

        # 2. Compute normalised quaternions
        normalised_quats = torch.nn.functional.normalize(x[:, 0:4], dim=1)  # Automatically avoids division by zero by using max(norm, epsilon) as the denominator.

        # 3. Compute quaternion orientation loss.
        orientation_loss = pose_algebra.vectorised_quat_distance(normalised_quats, y[:, 0:4])

        # 4. Compute translation norm, translation normalisation loss.
        transl_norms = torch.norm(x[:, 4:7], dim=1)
        translation_normalisation_loss = torch.mean(torch.abs(transl_norms - 1))

        # 5. Normalise translations.
        normalised_transl = torch.nn.functional.normalize(x[:, 4:7])  # Automatically avoids division by zero by using max(norm, epsilon) as the denominator.

        # 6. Normalise GT translations.
        gt_normalised_transl = torch.nn.functional.normalize(y[:, 4:7], dim=1)  # Automatically avoids division by zero by using max(norm, epsilon) as the denominator.

        # 7. Compute translation direction loss.
        translation_direction_loss = pose_algebra.vectorised_translation_direction_distance(normalised_transl, gt_normalised_transl)

        # 8. Compute full loss.
        loss = orientation_loss + translation_direction_loss + 0.5 * quaternion_normalisation_loss + 0.5 * translation_normalisation_loss

        return [loss, orientation_loss, translation_direction_loss, quaternion_normalisation_loss, translation_normalisation_loss]

