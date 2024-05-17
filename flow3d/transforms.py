from typing import Literal, overload

import roma
import torch
import torch.nn.functional as F


def rt_to_mat4(
    R: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        R (torch.Tensor): (..., 3, 3).
        t (torch.Tensor): (..., 3).
        s (torch.Tensor): (...,).

    Returns:
        torch.Tensor: (..., 4, 4)
    """
    mat34 = torch.cat([R, t[..., None]], dim=-1)
    if s is None:
        bottom = (
            mat34.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            .reshape((1,) * (mat34.dim() - 2) + (1, 4))
            .expand(mat34.shape[:-2] + (1, 4))
        )
    else:
        bottom = F.pad(1.0 / s[..., None, None], (3, 0), value=0.0)
    mat4 = torch.cat([mat34, bottom], dim=-2)
    return mat4


def rmat_to_cont_6d(matrix):
    """
    :param matrix (*, 3, 3)
    :returns 6d vector (*, 6)
    """
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont_6d_to_rmat(cont_6d):
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)


def solve_procrustes(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor | None = None,
    enforce_se3: bool = False,
    rot_type: Literal["quat", "mat", "6d"] = "quat",
):
    """
    Solve the Procrustes problem to align two point clouds, by solving the
    following problem:

    min_{s, R, t} || s * (src @ R.T + t) - dst ||_2, s.t. R.T @ R = I and det(R) = 1.

    Args:
        src (torch.Tensor): (N, 3).
        dst (torch.Tensor): (N, 3).
        weights (torch.Tensor | None): (N,), optional weights for alignment.
        enforce_se3 (bool): Whether to enforce the transfm to be SE3.

    Returns:
        sim3 (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            q (torch.Tensor): (4,), rotation component in quaternion of WXYZ
                format.
            t (torch.Tensor): (3,), translation component.
            s (torch.Tensor): (), scale component.
        error (torch.Tensor): (), average L2 distance after alignment.
    """
    # Compute weights.
    if weights is None:
        weights = src.new_ones(src.shape[0])
    weights = weights[:, None] / weights.sum()
    # Normalize point positions.
    src_mean = (src * weights).sum(dim=0)
    dst_mean = (dst * weights).sum(dim=0)
    src_cent = src - src_mean
    dst_cent = dst - dst_mean
    # Normalize point scales.
    if not enforce_se3:
        src_scale = (src_cent**2 * weights).sum(dim=-1).mean().sqrt()
        dst_scale = (dst_cent**2 * weights).sum(dim=-1).mean().sqrt()
    else:
        src_scale = dst_scale = src.new_tensor(1.0)
    src_scaled = src_cent / src_scale
    dst_scaled = dst_cent / dst_scale
    # Compute the matrix for the singular value decomposition (SVD).
    matrix = (weights * dst_scaled).T @ src_scaled
    U, _, Vh = torch.linalg.svd(matrix)
    # Special reflection case.
    S = torch.eye(3, device=src.device)
    if torch.det(U) * torch.det(Vh) < 0:
        S[2, 2] = -1
    R = U @ S @ Vh
    # Compute the transformation.
    if rot_type == "quat":
        rot = roma.rotmat_to_unitquat(R).roll(1, dims=-1)
    elif rot_type == "6d":
        rot = rmat_to_cont_6d(R)
    else:
        rot = R
    s = dst_scale / src_scale
    t = dst_mean / s - src_mean @ R.T
    sim3 = rot, t, s
    # Debug: error.
    procrustes_dst = torch.einsum(
        "ij,nj->ni", rt_to_mat4(R, t, s), F.pad(src, (0, 1), value=1.0)
    )
    procrustes_dst = procrustes_dst[:, :3] / procrustes_dst[:, 3:]
    error_before = (torch.linalg.norm(dst - src, dim=-1) * weights[:, 0]).sum()
    error = (torch.linalg.norm(dst - procrustes_dst, dim=-1) * weights[:, 0]).sum()
    # print(f"Procrustes error: {error_before} -> {error}")
    # if error_before < error:
    #     print("Something is wrong.")
    #     __import__("ipdb").set_trace()
    return sim3, (error.item(), error_before.item())
