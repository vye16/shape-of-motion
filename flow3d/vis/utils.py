import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from viser import ViserServer


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class VisManager(metaclass=Singleton):
    _servers = {}


def get_server(port: int | None = None) -> ViserServer:
    manager = VisManager()
    if port is None:
        avail_ports = list(manager._servers.keys())
        port = avail_ports[0] if len(avail_ports) > 0 else 8890
    if port not in manager._servers:
        manager._servers[port] = ViserServer(port=port, verbose=False)
    return manager._servers[port]


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


def make_video_divisble(
    video: torch.Tensor | np.ndarray, block_size=16
) -> torch.Tensor | np.ndarray:
    H, W = video.shape[1:3]
    H_new = H - H % block_size
    W_new = W - W % block_size
    return video[:, :H_new, :W_new]


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        matplotlib.colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor | None = None,
    near_plane: float | None = None,
    far_plane: float | None = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def apply_normal_colormap(normal: torch.Tensor) -> torch.Tensor:
    """Converts a normal image to color for easier analysis.

    Args:
        normal (torch.Tensor): (..., 3) float32 normal.

    Returns:
        (..., 3) colored normal image with colors in [0, 1].
    """
    normal = normal / normal.norm(dim=-1, keepdim=True)
    normal = torch.clip(normal, -1.0, 1.0)
    normal = (normal + 1.0) / 2.0
    return normal


def float2uint8(x):
    return (255.0 * x).astype(np.uint8)


def uint82float(img):
    return np.ascontiguousarray(img) / 255.0


def drawMatches(
    img1,
    img2,
    kp1,
    kp2,
    num_vis=200,
    center=None,
    idx_vis=None,
    radius=2,
    seed=1234,
    mask=None,
):
    num_pts = len(kp1)
    if idx_vis is None:
        if num_vis < num_pts:
            rng = np.random.RandomState(seed)
            idx_vis = rng.choice(num_pts, num_vis, replace=False)
        else:
            idx_vis = np.arange(num_pts)

    kp1_vis = kp1[idx_vis]
    kp2_vis = kp2[idx_vis]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    kp1_vis[:, 0] = np.clip(kp1_vis[:, 0], a_min=0, a_max=w1 - 1)
    kp1_vis[:, 1] = np.clip(kp1_vis[:, 1], a_min=0, a_max=h1 - 1)

    kp2_vis[:, 0] = np.clip(kp2_vis[:, 0], a_min=0, a_max=w2 - 1)
    kp2_vis[:, 1] = np.clip(kp2_vis[:, 1], a_min=0, a_max=h2 - 1)

    img1 = float2uint8(img1)
    img2 = float2uint8(img2)

    if center is None:
        center = np.median(kp1, axis=0)

    set_max = range(128)
    colors = {m: i for i, m in enumerate(set_max)}
    colors = {
        m: (255 * np.array(plt.cm.hsv(i / float(len(colors))))[:3][::-1]).astype(
            np.int32
        )
        for m, i in colors.items()
    }

    if mask is not None:
        ind = np.argsort(mask)[::-1]
        kp1_vis = kp1_vis[ind]
        kp2_vis = kp2_vis[ind]
        mask = mask[ind]

    for i, (pt1, pt2) in enumerate(zip(kp1_vis, kp2_vis)):
        # random_color = tuple(np.random.randint(low=0, high=255, size=(3,)).tolist())
        coord_angle = np.arctan2(pt1[1] - center[1], pt1[0] - center[0])
        corr_color = np.int32(64 * coord_angle / np.pi) % 128
        color = tuple(colors[corr_color].tolist())

        if (
            (pt1[0] <= w1 - 1)
            and (pt1[0] >= 0)
            and (pt1[1] <= h1 - 1)
            and (pt1[1] >= 0)
        ):
            img1 = cv2.circle(
                img1, (int(pt1[0]), int(pt1[1])), radius, color, -1, cv2.LINE_AA
            )
        if (
            (pt2[0] <= w2 - 1)
            and (pt2[0] >= 0)
            and (pt2[1] <= h2 - 1)
            and (pt2[1] >= 0)
        ):
            if mask is not None and mask[i]:
                continue
                # img2 = cv2.drawMarker(img2, (int(pt2[0]), int(pt2[1])), color, markerType=cv2.MARKER_CROSS,
                #                       markerSize=int(5*radius), thickness=int(radius/2), line_type=cv2.LINE_AA)
            else:
                img2 = cv2.circle(
                    img2, (int(pt2[0]), int(pt2[1])), radius, color, -1, cv2.LINE_AA
                )

    out = np.concatenate([img1, img2], axis=1)
    return out


def plot_correspondences(
    rgbs, kpts, query_id=0, masks=None, num_vis=1000000, radius=3, seed=1234
):
    num_rgbs = len(rgbs)
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(kpts.shape[1])
    kpts = kpts[:, permutation, :][:, :num_vis]
    if masks is not None:
        masks = masks[:, permutation][:, :num_vis]

    rgbq = rgbs[query_id]  # [h, w, 3]
    kptsq = kpts[query_id]  # [n, 2]

    frames = []
    for i in range(num_rgbs):
        rgbi = rgbs[i]
        kptsi = kpts[i]
        if masks is not None:
            maski = masks[i]
        else:
            maski = None
        frame = drawMatches(
            rgbq,
            rgbi,
            kptsq,
            kptsi,
            mask=maski,
            num_vis=num_vis,
            radius=radius,
            seed=seed,
        )
        frames.append(frame)
    return frames
