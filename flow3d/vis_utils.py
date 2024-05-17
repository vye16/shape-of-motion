import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def project_2d_tracks(tracks_3d_w, Ks, T_cw, return_depth=False):
    """
    :param tracks_3d_w (torch.Tensor): (T, N, 3)
    :param Ks (torch.Tensor): (T, 3, 3)
    :param T_cw (torch.Tensor): (T, 4, 4)
    :returns tracks_2d (torch.Tensor): (T, N, 2)
    """
    tracks_3d_c = torch.einsum(
        "tij,tnj->tni", T_cw, F.pad(tracks_3d_w, (0, 1), value=1)
    )[..., :3]
    tracks_3d_v = torch.einsum("tij,tnj->tni", Ks, tracks_3d_c)
    if return_depth:
        return (
            tracks_3d_v[..., :2] / torch.clamp(tracks_3d_v[..., 2:], min=1e-5),
            tracks_3d_v[..., 2],
        )
    return tracks_3d_v[..., :2] / torch.clamp(tracks_3d_v[..., 2:], min=1e-5)


def draw_keypoints_video(
    imgs, kps, colors=None, occs=None, cmap: str = "gist_rainbow", radius: int = 3
):
    """
    :param imgs (np.ndarray): (T, H, W, 3) uint8 [0, 255]
    :param kps (np.ndarray): (N, T, 2)
    :param colors (np.ndarray): (N, 3) float [0, 1]
    :param occ (np.ndarray): (N, T) bool
    return out_frames (T, H, W, 3)
    """
    if colors is None:
        label = np.linspace(0, 1, kps.shape[0])
        colors = np.asarray(plt.get_cmap(cmap)(label))[..., :3]
    out_frames = []
    for t in range(len(imgs)):
        occ = occs[:, t] if occs is not None else None
        vis = draw_keypoints_cv2(imgs[t], kps[:, t], colors, occ, radius=radius)
        out_frames.append(vis)
    return out_frames


def draw_keypoints_cv2(img, kps, colors=None, occs=None, radius=3):
    """
    :param img (H, W, 3)
    :param kps (N, 2)
    :param occs (N)
    :param colors (N, 3) from 0 to 1
    """
    out_img = img.copy()
    kps = kps.round().astype("int").tolist()
    if colors is not None:
        colors = (255 * colors).astype("int").tolist()
    for n in range(len(kps)):
        kp = kps[n]
        color = colors[n] if colors is not None else (255, 0, 0)
        thickness = -1 if occs is None or occs[n] == 0 else 1
        out_img = cv2.circle(out_img, kp, radius, color, thickness, cv2.LINE_AA)
    return out_img
