from typing import List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple


class SceneNormDict(TypedDict):
    scale: float
    transfm: torch.Tensor


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [to_device(v, device) for v in batch]
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def normalize_coords(coords, h, w):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [-inf, inf], np.float32
      expected_dist:, [-inf, inf], np.float32

    Returns:
      visibles: bool
    """

    def sigmoid(x):
        if x.dtype == np.ndarray:
            return 1 / (1 + np.exp(-x))
        else:
            return torch.sigmoid(x)

    visibles = (1 - sigmoid(occlusions)) * (1 - sigmoid(expected_dist)) > 0.5
    return visibles


def parse_tapir_track_info(occlusions, expected_dist):
    """
    return:
        valid_visible: mask of visible & confident points
        valid_invisible: mask of invisible & confident points
        confidence: clamped confidence scores (all < 0.5 -> 0)
    """
    visiblility = 1 - F.sigmoid(occlusions)
    confidence = 1 - F.sigmoid(expected_dist)
    valid_visible = visiblility * confidence > 0.5
    valid_invisible = (1 - visiblility) * confidence > 0.5
    # set all confidence < 0.5 to 0
    confidence = confidence * (valid_visible | valid_invisible).float()
    return valid_visible, valid_invisible, confidence


def get_tracks_3d_for_query_frame(
    query_index: int,
    query_img: torch.Tensor,
    tracks_2d: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    inv_Ks: torch.Tensor,
    c2ws: torch.Tensor,
):
    """
    :param query_index (int)
    :param query_img [H, W, 3]
    :param tracks_2d [N, T, 4]
    :param depths [T, H, W]
    :param masks [T, H, W]
    :param inv_Ks [T, 3, 3]
    :param c2ws [T, 4, 4]
    returns (
        tracks_3d [N, T, 3]
        track_colors [N, 3]
        visibles [N, T]
        invisibles [N, T]
        confidences [N, T]
    )
    """
    T, H, W = depths.shape
    query_img = query_img[None].permute(0, 3, 1, 2)  # (1, 3, H, W)
    tracks_2d = tracks_2d.swapaxes(0, 1)  # (T, N, 4)
    tracks_2d, occs, dists = (
        tracks_2d[..., :2],
        tracks_2d[..., 2],
        tracks_2d[..., 3],
    )
    # visibles = postprocess_occlusions(occs, dists)
    # (T, N), (T, N), (T, N)
    visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
    # Unproject 2D tracks to 3D.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    track_depths = F.grid_sample(
        depths[:, None],
        normalize_coords(tracks_2d[:, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[:, 0, 0]
    tracks_3d = (
        torch.einsum(
            "nij,npj->npi",
            inv_Ks,
            F.pad(tracks_2d, (0, 1), value=1.0),
        )
        * track_depths[..., None]
    )
    tracks_3d = torch.einsum("nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0))[
        ..., :3
    ]
    # Filter out out-of-mask tracks.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    is_in_masks = (
        F.grid_sample(
            masks[:, None],
            normalize_coords(tracks_2d[:, None], H, W),
            align_corners=True,
        )[:, 0, 0]
        == 1
    )
    visibles *= is_in_masks
    invisibles *= is_in_masks
    confidences *= is_in_masks.float()

    # valid if in the fg mask at least 40% of the time
    # in_mask_counts = is_in_masks.sum(0)
    # t = 0.25
    # thresh = min(t * T, in_mask_counts.float().quantile(t).item())
    # valid = in_mask_counts > thresh
    valid = is_in_masks[query_index]
    # valid if visible 5% of the time
    visible_counts = visibles.sum(0)
    valid = valid & (
        visible_counts
        >= min(
            int(0.05 * T),
            visible_counts.float().quantile(0.1).item(),
        )
    )

    # Get track's color from the query frame.
    # (1, 3, H, W), (1, 1, N, 2) -> (1, 3, 1, N) -> (N, 3)
    track_colors = F.grid_sample(
        query_img,
        normalize_coords(tracks_2d[query_index : query_index + 1, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[0, :, 0].T
    return (
        tracks_3d[:, valid].swapdims(0, 1),
        track_colors[valid],
        visibles[:, valid].swapdims(0, 1),
        invisibles[:, valid].swapdims(0, 1),
        confidences[:, valid].swapdims(0, 1),
    )


def _get_padding(x, k, stride, padding, same: bool):
    if same:
        ih, iw = x.size()[2:]
        if ih % stride[0] == 0:
            ph = max(k[0] - stride[0], 0)
        else:
            ph = max(k[0] - (ih % stride[0]), 0)
        if iw % stride[1] == 0:
            pw = max(k[1] - stride[1], 0)
        else:
            pw = max(k[1] - (iw % stride[1]), 0)
        pl = pw // 2
        pr = pw - pl
        pt = ph // 2
        pb = ph - pt
        padding = (pl, pr, pt, pb)
    else:
        padding = padding
    return padding


def median_filter_2d(x, kernel_size=3, stride=1, padding=1, same: bool = True):
    """
    :param x [B, C, H, W]
    """
    k = _pair(kernel_size)
    stride = _pair(stride)  # convert to tuple
    padding = _quadruple(padding)  # convert to l, r, t, b
    # using existing pytorch functions and tensor ops so that we get autograd,
    # would likely be more efficient to implement from scratch at C/Cuda level
    x = F.pad(x, _get_padding(x, k, stride, padding, same), mode="reflect")
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


def masked_median_blur(image, mask, kernel_size=11):
    """
    Args:
        image: [B, C, H, W]
        mask: [B, C, H, W]
        kernel_size: int
    """
    assert image.shape == mask.shape
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {image.shape}")

    padding: Tuple[int, int] = _compute_zero_padding((kernel_size, kernel_size))

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d((kernel_size, kernel_size)).to(image)
    b, c, h, w = image.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(
        image.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    masks: torch.Tensor = F.conv2d(
        mask.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1
    )
    features = features.view(b, c, -1, h, w).permute(
        0, 1, 3, 4, 2
    )  # BxCxxHxWx(K_h * K_w)
    min_value, max_value = features.min(), features.max()
    masks = masks.view(b, c, -1, h, w).permute(0, 1, 3, 4, 2)  # BxCxHxWx(K_h * K_w)
    index_invalid = (1 - masks).nonzero(as_tuple=True)
    index_b, index_c, index_h, index_w, index_k = index_invalid
    features[(index_b[::2], index_c[::2], index_h[::2], index_w[::2], index_k[::2])] = (
        min_value
    )
    features[
        (index_b[1::2], index_c[1::2], index_h[1::2], index_w[1::2], index_k[1::2])
    ] = max_value
    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=-1)[0]

    return median


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def get_binary_kernel2d(
    window_size: tuple[int, int] | int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    from kornia
    Create a binary kernel to extract the patches.
    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    ky, kx = _unpack_2d_ks(window_size)

    window_range = kx * ky

    kernel = torch.zeros((window_range, window_range), device=device, dtype=dtype)
    idx = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, "2D Kernel size should have a length of 2."
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


## Functions from GaussianShader.
def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (
        W - 1
    )
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (
        H - 1
    )
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x, indexing="ij")
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(
        B, N, C, H, W, 3
    )  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)  # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz


def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(
        depth_image[None, None, None, ...], intrinsic_matrix[None, ...]
    )
    xyz_cam = xyz_cam.reshape(-1, 3)
    xyz_world = torch.cat(
        [xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], dim=-1
    ) @ torch.inverse(extrinsic_matrix).transpose(0, 1)
    xyz_world = xyz_world[..., :3]

    return xyz_world


def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal


def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix)  # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal
