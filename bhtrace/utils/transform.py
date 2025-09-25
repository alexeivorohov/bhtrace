import torch

def sph2cart(coords):
    """
    Converts a batch of spherical coordinates (t, r, theta, phi) to Cartesian (t, x, y, z).
    """
    t, r, theta, phi = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([t, x, y, z], dim=-1)

def cart2sph(coords):
    """
    Converts a batch of Cartesian coordinates (t, x, y, z) to spherical (t, r, theta, phi).
    """
    t, x, y, z = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
    x2y2 = x**2 + y**2
    r = torch.sqrt(x2y2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.arctan2(y, x)
    return torch.stack([t, r, theta, phi], dim=-1)

def EulerRotation(
        X: torch.Tensor,
        dphi: float | torch.Tensor,
        dth: float | torch.Tensor
        ):
    '''
    Euler rotation of unit vector(s) in cartesian coordinates

    Inputs:
    - X: torch.Tensor - vector(s) 
    - dphi: float | torch.Tensor - azimuthal rotation
    - dth: float | torch.Tensor - ... rotation

    Outputs:
    - outX: torch.Tensor - rotated vector
    '''

    cdphi = torch.cos(dphi)
    sdphi = torch.sin(dphi)

    sdth = torch.sin(dth)
    cdth = torch.cos(dth)

    outX0 = (X[..., 0]*cdphi - X[..., 1]*sdphi)*cdth - X[..., 0]*X[..., 2]*sdth
    outX1 = (X[..., 0]*sdphi + X[..., 1]*cdphi)*cdth - X[..., 1]*X[..., 2]*sdth
    outX2 = torch.pow((X[..., 0]**2 + X[..., 1]**2), -0.5)*sdth + X[..., 2]*cdth

    return torch.stack([outX0, outX1, outX2], -1)


def rotate_points_cloud(points, dir_a, dir_b, gamma=0.0):
    """
    Rotate a point cloud so that dir_a aligns with dir_b, then rotate around dir_b by angle gamma.

    Args:
        points (torch.Tensor): Nx3 tensor of points.
        dir_a (torch.Tensor): Original orientation vector, shape (3,).
        dir_b (torch.Tensor): Target orientation vector, shape (3,).
        gamma (float): Additional rotation angle (in radians) around dir_b.

    Returns:
        torch.Tensor: Rotated point cloud Nx3.
    """
    # Normalize input vectors
    v1 = dir_a / dir_a.norm()
    v2 = dir_b / dir_b.norm()

    # Compute rotation axis and angle to align v1 to v2
    axis = torch.cross(v1, v2)
    cos_angle = torch.dot(v1, v2)

    # Handle parallel and anti-parallel cases
    if torch.allclose(axis, torch.zeros(3), atol=1e-6):
        if cos_angle > 0:
            # No rotation needed
            R_align = torch.eye(3)
        else:
            # 180 degree rotation around any orthogonal axis
            orthogonal_axis = torch.tensor([1., 0., 0.])
            if torch.allclose(v1, orthogonal_axis, atol=1e-6):
                orthogonal_axis = torch.tensor([0., 1., 0.])
            axis = torch.cross(v1, orthogonal_axis)
            axis = axis / axis.norm()
            angle_align = torch.pi
            R_align = rotation_matrix(axis, angle_align)
    else:
        axis = axis / axis.norm()
        angle_align = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        R_align = rotation_matrix(axis, angle_align)

    # Rotation matrix around dir_b by 'angle'
    axis_b = v2
    R_rotate = rotation_matrix(axis_b, gamma)

    # Combined rotation: first align, then rotate around dir_b
    R = R_rotate @ R_align

    # Rotate points
    rotated_points = points @ R.T
    return rotated_points


def rotation_matrix(axis: torch.Tensor, angle: float):
    """
    Rodrigues' rotation formula for rotation matrix around a given axis.

    Args:
        axis (torch.Tensor): Rotation axis (3,).
        angle (float): Rotation angle in radians.

    Returns:
        torch.Tensor: 3x3 rotation matrix.
    """
    angle = torch.Tensor([angle])
    axis = axis / axis.norm()
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=axis.dtype)

    I = torch.eye(3, dtype=axis.dtype)
    R = I + torch.sin(angle)*K + (1 - torch.cos(angle))*(K @ K)
    return R
