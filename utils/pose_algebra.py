"""
Name: pose_algebra.py
Description: Contains functions used to work on data representing poses, such as transformation matrices and quaternions.
-----
Authors: Matteo Taiana, Matteo Toso.
Licence: MIT. Copyright 2022, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import torch


def quat_mul(q, r):
    """Quaternion multiplication. Input variables have to be Torch tensors.

       Multiply quaternion(s) q with quaternion(s) r.
       Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
       Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def quat_distance(q0, q1):
    # This function seems to be safe for the case in which q0=[0,0,0,0], it returns pi, which is the worst possible result.
    # And it returns 0 for the case in which q0==q1.
    # Formula for computing distance: https://it.mathworks.com/help/fusion/ref/quaternion.dist.html
    # angularDistance = 2*acos(abs(parts(p*conj(q))));
    # θ_z=2cos^-1(real(z)), with z = product of q0, conj(q1)
    # First element of the quaternion is the real one.
    # Conjugate quaternion has the sign of the imaginary parts inverted.
    # The product is the quat product.
    conj_q1 = torch.tensor([q1[0], -q1[1], -q1[2], -q1[3]], device='cuda')
    z = quat_mul(q0, conj_q1)
    # z might happen to be outside the valid input range of acos, leading to NaN's, so we make sure it is in the range.
    z[0] = max(-1, z[0])
    z[0] = min(1, z[0])
    angle_rad = 2*torch.acos(torch.abs(z[0]))
    return angle_rad


def vectorised_quat_distance(q0s, q1s):
    # This function seems to be safe for the case in which q0=[0,0,0,0], it returns pi, which is the worst possible result.
    # And it returns 0 for the case in which q0==q1.
    # Formula for computing distance: https://it.mathworks.com/help/fusion/ref/quaternion.dist.html
    # angularDistance = 2*acos(abs(parts(p*conj(q))));
    # θ_z=2cos^-1(real(z)), with z = product of q0, conj(q1)
    # First element of the quaternion is the real one.
    # Conjugate quaternion has the sign of the imaginary parts inverted.
    # The product is the quat product.
    z = quat_mul(q0s, torch.cat((q1s[:, 0].view(-1, 1), -q1s[:, 1:4]), dim=1))

    # z might happen to be outside the valid input range of acos, leading to NaN's, so we make sure it is in the range.
    epsilon = 1e-7  # This might be needed because acos() of values too close to -1 or +1 makes the gradient diverge -> NaN's!
                    # See: https://github.com/pytorch/pytorch/issues/8069
    clamped_z = torch.clamp(z[:, 0], min=-1+epsilon, max=1-epsilon)

    angle_rad = torch.mean(2 * torch.acos(torch.abs(clamped_z)))
    return angle_rad


def vectorised_translation_direction_distance(t0s, t1s):
    batch_size = t0s.shape[0]
    t0s = t0s.reshape(batch_size, 1, 3)
    t1s = t1s.reshape(batch_size, 3, 1)
    cosines = torch.matmul(t0s, t1s).squeeze(1)

    # Make sure cosines are in the (-1, +1) range, WITH A LITTLE MARGIN.
    epsilon = 1e-7  # This might be needed because acos() of values too close to -1 or +1 makes the gradient diverge -> NaN's!
                    # See: https://github.com/pytorch/pytorch/issues/8069
    clamped_cosines = torch.clamp(cosines, min=-1+epsilon, max=1-epsilon)

    angle_rad = torch.mean(torch.arccos(clamped_cosines))
    return angle_rad


def quaternion_from_matrix_torch(R, device):
    """Code inspired from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/.
        Unit quaternion to represent rotation: (w, x, y, z).
    """
    epsilon = 0.00001
    q = torch.zeros(4, device=device)

    trace = torch.trace(R)
    if trace > 0.0:
        sqrt_trace = torch.sqrt(1.0 + trace)
        q[0] = 0.5 * sqrt_trace
        q[1] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
        q[2] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
        q[3] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            sqrt_trace = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            if sqrt_trace == 0:
                sqrt_trace = epsilon
            q[0] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
            q[1] = 0.5 * sqrt_trace
            q[2] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[3] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
        elif R[1, 1] > R[2, 2]:
            sqrt_trace = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            if sqrt_trace == 0:
                sqrt_trace = epsilon
            q[0] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
            q[1] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[2] = 0.5 * sqrt_trace
            q[3] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
        else:
            sqrt_trace = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
            q[1] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
            q[2] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
            q[3] = 0.5 * sqrt_trace
    return q


def matrix_from_quaternion_torch(q, device):
    """Formula as seen in: http://www.songho.ca/opengl/gl_quaternion.html."""
    q_norm = torch.norm(q)
    if q_norm == 0:
      uq = q
    else:
      uq = q / q_norm

    w, x, y, z = uq
    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    R = torch.tensor([[1.0 - y2 - z2, xy - zw, xz + yw],
                      [xy + zw, 1.0 - x2 - z2, yz - xw],
                      [xz - yw, yz + xw, 1.0 - x2 - y2]], device=device)

    return R


def pose_matrix_to_pose_quaternion(pose_matrix, device):
    """The name 'pose quaternion' is loose in this case, for this function we mean the whole transformation, including rotation and
       translation.
    """
    q = quaternion_from_matrix_torch(pose_matrix[0:3, 0:3], device=device)
    t = pose_matrix[0:3, 3]
    quat_and_translation = torch.cat((q, t))
    return quat_and_translation


def pose_quaternion_to_pose_matrix(pose_quat, device):
    """The name 'pose quaternion' loose in this case, for this function we mean the whole transformation, including rotation and
       translation.
    """
    R = matrix_from_quaternion_torch(pose_quat[0:4], device=device)
    pose_matrix = torch.cat((torch.cat((R, pose_quat[4:7].view(3, 1)), dim=1), torch.tensor((0, 0, 0, 1), device=device).view(1, 4)))
    return pose_matrix


def invert_pose_quaternion_vectorised(pose_quats):
    """The name 'pose quaternion' is loose in this case, for this function we mean the whole transformation, including rotation and
       translation.
    """
    n_quats = pose_quats.shape[0]
    qs = pose_quats[:, 0:4]
    inverse_rotations = pose_quats[:, 0:4].clone()
    inverse_rotations[:, 1:4] = -inverse_rotations[:, 1:4]
    epsilon = 0.00001
    quat_norms = torch.norm(qs, dim=1) + epsilon

    t = torch.cat((torch.zeros((n_quats, 1), device='cuda'), -pose_quats[:, 4:7]), dim=1)
    inverse_translations = quat_mul(inverse_rotations/quat_norms.view(n_quats, 1), quat_mul(t, qs/quat_norms.view(n_quats, 1)))  # See non-vectorised function for an explanation of what is done here.


    # If the quaternion is a unit quaternion, the first element of the resulting quaternion has to be close to 0.
    # We could put an assertion here.
    inverse_quat_poses = torch.cat((inverse_rotations, inverse_translations[:, 1:4]), dim=1)
    return inverse_quat_poses



def compute_relative_pose_quaternion(dest_pose, source_pose, device):
    # Currently (Feb 2022) assumes that the two transformation matrices are active:
    # they transform points from the camera reference frame to the world reference frame.
    # They contain the pose of the camera as seen from the world.
    T = torch.matmul(torch.inverse(dest_pose), source_pose)    # This version should be used when the transformation matrices are active.

    # Express the relative transformation in terms of a quaternion and a translation.
    relative_pose = pose_matrix_to_pose_quaternion(T, device=device)
    return relative_pose


