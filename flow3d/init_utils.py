import time
import cupy as cp
from cuml import HDBSCAN, KMeans
import imageio.v3 as iio
from matplotlib.pyplot import get_cmap
import numpy as np

# from pytorch3d.ops import sample_farthest_points
import roma
import torch
from tqdm import tqdm
from loguru import logger as guru
import torch.nn.functional as F
from typing import Literal

from viser import ViserServer
from flow3d.loss_utils import (
    compute_accel_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
    get_weights_for_procrustes,
    knn,
)
from flow3d.params import GaussianParams, MotionBases
from flow3d.tensor_dataclass import (
    StaticObservations,
    TrackObservations,
)
from flow3d.transforms import cont_6d_to_rmat, rt_to_mat4, solve_procrustes
from flow3d.vis.utils import (
    draw_keypoints_video,
    get_server,
    project_2d_tracks,
)


def init_fg_from_tracks_3d(
    cano_t: int, tracks_3d: TrackObservations, motion_coefs: torch.Tensor
) -> GaussianParams:
    """
    using dataclasses individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_fg = tracks_3d.xyz.shape[0]

    # Initialize gaussian colors.
    colors = torch.logit(tracks_3d.colors)
    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(tracks_3d.xyz[:, cano_t], 3)
    dists = torch.from_numpy(dists)
    scales = dists.mean(dim=-1, keepdim=True)
    scales = scales.clamp(torch.quantile(scales, 0.05), torch.quantile(scales, 0.95))
    scales = torch.log(scales.repeat(1, 3))
    # Initialize gaussian means.
    means = tracks_3d.xyz[:, cano_t]
    # Initialize gaussian orientations as random.
    quats = torch.rand(num_fg, 4)
    # Initialize gaussian opacities.
    opacities = torch.logit(torch.full((num_fg,), 0.7))
    gaussians = GaussianParams(means, quats, scales, colors, opacities, motion_coefs)
    return gaussians


def init_bg(
    points: StaticObservations,
) -> GaussianParams:
    """
    using dataclasses instead of individual tensors so we know they're consistent
    and are always masked/filtered together
    """
    num_init_bg_gaussians = points.xyz.shape[0]
    bg_scene_center = points.xyz.mean(0)
    bg_points_centered = points.xyz - bg_scene_center
    bg_min_scale = bg_points_centered.quantile(0.05, dim=0)
    bg_max_scale = bg_points_centered.quantile(0.95, dim=0)
    bg_scene_scale = torch.max(bg_max_scale - bg_min_scale).item() / 2.0
    bkdg_colors = torch.logit(points.colors)

    # Initialize gaussian scales: find the average of the three nearest
    # neighbors in the first frame for each point and use that as the
    # scale.
    dists, _ = knn(points.xyz, 3)
    dists = torch.from_numpy(dists)
    bg_scales = dists.mean(dim=-1, keepdim=True)
    bkdg_scales = torch.log(bg_scales.repeat(1, 3))

    bg_means = points.xyz

    # Initialize gaussian orientations by normals.
    local_normals = points.normals.new_tensor([[0.0, 0.0, 1.0]]).expand_as(
        points.normals
    )
    bg_quats = roma.rotvec_to_unitquat(
        F.normalize(local_normals.cross(points.normals), dim=-1)
        * (local_normals * points.normals).sum(-1, keepdim=True).acos_()
    ).roll(1, dims=-1)
    bg_opacities = torch.logit(torch.full((num_init_bg_gaussians,), 0.7))
    gaussians = GaussianParams(
        bg_means,
        bg_quats,
        bkdg_scales,
        bkdg_colors,
        bg_opacities,
        scene_center=bg_scene_center,
        scene_scale=bg_scene_scale,
    )
    return gaussians


def init_motion_params_with_procrustes(
    tracks_3d: TrackObservations,
    num_bases: int,
    rot_type: Literal["quat", "6d"],
    cano_t: int,
    cluster_init_method: str = "kmeans",
    min_mean_weight: float = 0.1,
    vis: bool = False,
    port: int = 8890,
) -> tuple[MotionBases, torch.Tensor, TrackObservations]:
    device = tracks_3d.xyz.device
    num_frames = tracks_3d.xyz.shape[1]
    # sample centers and get initial se3 motion bases by solving procrustes
    means_cano = tracks_3d.xyz[:, cano_t].clone()  # [num_gaussians, 3]

    # remove outliers
    scene_center = means_cano.median(dim=0).values
    print(f"{scene_center=}")
    dists = torch.norm(means_cano - scene_center, dim=-1)
    dists_th = torch.quantile(dists, 0.95)
    valid_mask = dists < dists_th

    # remove tracks that are not visible in any frame
    valid_mask = valid_mask & tracks_3d.visibles.any(dim=1)
    print(f"{valid_mask.sum()=}")

    tracks_3d = tracks_3d.filter_valid(valid_mask)

    if vis:
        server = get_server(port)
        try:
            pts = tracks_3d.xyz.cpu().numpy()
            clrs = tracks_3d.colors.cpu().numpy()
            while True:
                for t in range(num_frames):
                    server.scene.add_point_cloud("points", pts[:, t], clrs)
                    time.sleep(0.3)
        except KeyboardInterrupt:
            pass

    means_cano = means_cano[valid_mask]

    sampled_centers, num_bases, labels = sample_initial_bases_centers(
        cluster_init_method, cano_t, tracks_3d, num_bases
    )

    # assign each point to the label to compute the cluster weight
    ids, counts = labels.unique(return_counts=True)
    ids = ids[counts > 100]
    num_bases = len(ids)
    sampled_centers = sampled_centers[:, ids]
    print(f"{num_bases=} {sampled_centers.shape=}")

    # compute basis weights from the distance to the cluster centers
    dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1)
    motion_coefs = 10 * torch.exp(-dists2centers)

    init_rots, init_ts = [], []

    if rot_type == "quat":
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        rot_dim = 4
    else:
        id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
        rot_dim = 6

    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)
    errs_before = np.full((num_bases, num_frames), -1.0)
    errs_after = np.full((num_bases, num_frames), -1.0)

    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames))
    print(f"{tgt_ts=}")
    for n, cluster_id in enumerate(ids):
        mask_in_cluster = labels == cluster_id
        cluster = tracks_3d.xyz[mask_in_cluster].transpose(
            0, 1
        )  # [num_frames, n_pts, 3]
        visibilities = tracks_3d.visibles[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        confidences = tracks_3d.confidences[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        weights = get_weights_for_procrustes(cluster, visibilities)
        prev_t = cano_t
        for cur_t in tgt_ts:
            # compute pairwise transform from cano_t
            procrustes_weights = (
                weights[cano_t]
                * weights[cur_t]
                * (confidences[cano_t] + confidences[cur_t])
                / 2
            )
            if procrustes_weights.sum() < min_mean_weight * num_frames:
                init_rots[n, cur_t] = init_rots[n, prev_t]
                init_ts[n, cur_t] = init_ts[n, prev_t]
                print(f"skipping {cur_t=} {procrustes_weights.sum()=}")
            else:
                se3, (err, err_before) = solve_procrustes(
                    cluster[cano_t],
                    cluster[cur_t],
                    weights=procrustes_weights,
                    enforce_se3=True,
                    rot_type=rot_type,
                )
                init_rot, init_t, _ = se3
                assert init_rot.shape[-1] == rot_dim
                # double cover
                if rot_type == "quat" and torch.linalg.norm(
                    init_rot - init_rots[n][prev_t]
                ) > torch.linalg.norm(-init_rot - init_rots[n][prev_t]):
                    init_rot = -init_rot
                init_rots[n, cur_t] = init_rot
                init_ts[n, cur_t] = init_t
                if err == np.nan:
                    print(f"{cur_t=} {err=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                if err_before == np.nan:
                    print(f"{cur_t=} {err_before=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                errs_after[n, cur_t] = err
                errs_before[n, cur_t] = err_before
            prev_t = cur_t

    guru.info(
        "procrustes init median error: {:.5f} => {:.5f}".format(
            np.median(errs_before[errs_before > 0]),
            np.median(errs_after[errs_after > 0]),
        )
    )
    guru.info(
        "procrustes init mean error: {:.5f} => {:.5f}".format(
            np.mean(errs_before[errs_before > 0]), np.mean(errs_after[errs_after > 0])
        )
    )
    guru.info(f"{init_rots.shape=}, {init_ts.shape=}, {motion_coefs.shape=}")

    if vis:
        server = get_server(port)
        center_idcs = torch.argmin(dists2centers, dim=0)
        print(f"{dists2centers.shape=} {center_idcs.shape=}")
        vis_se3_init_3d(server, init_rots, init_ts, means_cano[center_idcs])
        vis_tracks_3d(server, tracks_3d.xyz[center_idcs].numpy(), name="center_tracks")
        import ipdb

        ipdb.set_trace()

    bases = MotionBases(init_rots, init_ts)
    return bases, motion_coefs, tracks_3d


def run_initial_optim(
    fg: GaussianParams,
    bases: MotionBases,
    tracks_3d: TrackObservations,
    Ks: torch.Tensor,
    w2cs: torch.Tensor,
    num_iters: int = 1000,
    use_depth_range_loss: bool = False,
):
    """
    :param motion_rots: [num_bases, num_frames, 4|6]
    :param motion_transls: [num_bases, num_frames, 3]
    :param motion_coefs: [num_bases, num_frames]
    :param means: [num_gaussians, 3]
    """
    optimizer = torch.optim.Adam(
        [
            {"params": bases.params["rots"], "lr": 1e-2},
            {"params": bases.params["transls"], "lr": 3e-2},
            {"params": fg.params["motion_coefs"], "lr": 1e-2},
            {"params": fg.params["means"], "lr": 1e-3},
        ],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / num_iters)
    )
    G = fg.params.means.shape[0]
    num_frames = bases.num_frames
    device = bases.params["rots"].device

    w_smooth_func = lambda i, min_v, max_v, th: (
        min_v if i <= th else (max_v - min_v) * (i - th) / (num_iters - th) + min_v
    )

    gt_2d, gt_depth = project_2d_tracks(
        tracks_3d.xyz.swapaxes(0, 1), Ks, w2cs, return_depth=True
    )
    # (G, T, 2)
    gt_2d = gt_2d.swapaxes(0, 1)
    # (G, T)
    gt_depth = gt_depth.swapaxes(0, 1)

    ts = torch.arange(0, num_frames, device=device)
    ts_clamped = torch.clamp(ts, min=1, max=num_frames - 2)
    ts_neighbors = torch.cat((ts_clamped - 1, ts_clamped, ts_clamped + 1))  # i (3B,)

    pbar = tqdm(range(0, num_iters))
    for i in pbar:
        coefs = fg.get_coefs()
        transfms = bases.compute_transforms(ts, coefs)
        positions = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(fg.params["means"], (0, 1), value=1.0),
        )

        loss = 0.0
        track_3d_loss = masked_l1_loss(
            positions,
            tracks_3d.xyz,
            (tracks_3d.visibles.float() * tracks_3d.confidences)[..., None],
        )
        loss += track_3d_loss * 1.0

        pred_2d, pred_depth = project_2d_tracks(
            positions.swapaxes(0, 1), Ks, w2cs, return_depth=True
        )
        pred_2d = pred_2d.swapaxes(0, 1)
        pred_depth = pred_depth.swapaxes(0, 1)

        loss_2d = (
            masked_l1_loss(
                pred_2d,
                gt_2d,
                (tracks_3d.invisibles.float() * tracks_3d.confidences)[..., None],
                quantile=0.95,
            )
            / Ks[0, 0, 0]
        )
        loss += 0.5 * loss_2d

        if use_depth_range_loss:
            near_depths = torch.quantile(gt_depth, 0.0, dim=0, keepdim=True)
            far_depths = torch.quantile(gt_depth, 0.98, dim=0, keepdim=True)
            loss_depth_in_range = 0
            if (pred_depth < near_depths).any():
                loss_depth_in_range += (near_depths - pred_depth)[
                    pred_depth < near_depths
                ].mean()
            if (pred_depth > far_depths).any():
                loss_depth_in_range += (pred_depth - far_depths)[
                    pred_depth > far_depths
                ].mean()

            loss += loss_depth_in_range * w_smooth_func(i, 0.05, 0.5, 400)

        motion_coef_sparse_loss = 1 - (coefs**2).sum(dim=-1).mean()
        loss += motion_coef_sparse_loss * 0.01

        # motion basis should be smooth.
        w_smooth = w_smooth_func(i, 0.01, 0.1, 400)
        small_acc_loss = compute_se3_smoothness_loss(
            bases.params["rots"], bases.params["transls"]
        )
        loss += small_acc_loss * w_smooth

        small_acc_loss_tracks = compute_accel_loss(positions)
        loss += small_acc_loss_tracks * w_smooth * 0.5

        transfms_nbs = bases.compute_transforms(ts_neighbors, coefs)
        means_nbs = torch.einsum(
            "pnij,pj->pni", transfms_nbs, F.pad(fg.params["means"], (0, 1), value=1.0)
        )  # (G, 3n, 3)
        means_nbs = means_nbs.reshape(means_nbs.shape[0], 3, -1, 3)  # [G, 3, n, 3]
        z_accel_loss = compute_z_acc_loss(means_nbs, w2cs)
        loss += z_accel_loss * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            f"{loss.item():.3f} "
            f"{track_3d_loss.item():.3f} "
            f"{motion_coef_sparse_loss.item():.3f} "
            f"{small_acc_loss.item():.3f} "
            f"{small_acc_loss_tracks.item():.3f} "
            f"{z_accel_loss.item():.3f} "
        )


def random_quats(N: int) -> torch.Tensor:
    u = torch.rand(N, 1)
    v = torch.rand(N, 1)
    w = torch.rand(N, 1)
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * np.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * np.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * np.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * np.pi * w),
        ],
        -1,
    )
    return quats


def compute_means(ts, fg: GaussianParams, bases: MotionBases):
    transfms = bases.compute_transforms(ts, fg.get_coefs())
    means = torch.einsum(
        "pnij,pj->pni",
        transfms,
        F.pad(fg.params["means"], (0, 1), value=1.0),
    )
    return means


def vis_init_params(
    server,
    fg: GaussianParams,
    bases: MotionBases,
    name="init_params",
    num_vis: int = 100,
):
    idcs = np.random.choice(fg.num_gaussians, num_vis)
    labels = np.linspace(0, 1, num_vis)
    ts = torch.arange(bases.num_frames, device=bases.params["rots"].device)
    with torch.no_grad():
        pred_means = compute_means(ts, fg, bases)
        vis_means = pred_means[idcs].detach().cpu().numpy()
    vis_tracks_3d(server, vis_means, labels, name=name)


@torch.no_grad()
def vis_se3_init_3d(server, init_rots, init_ts, basis_centers):
    """
    :param init_rots: [num_bases, num_frames, 4|6]
    :param init_ts: [num_bases, num_frames, 3]
    :param basis_centers: [num_bases, 3]
    """
    # visualize the initial centers across time
    rot_dim = init_rots.shape[-1]
    assert rot_dim in [4, 6]
    num_bases = init_rots.shape[0]
    assert init_ts.shape[0] == num_bases
    assert basis_centers.shape[0] == num_bases
    labels = np.linspace(0, 1, num_bases)
    if rot_dim == 4:
        quats = F.normalize(init_rots, dim=-1, p=2)
        rmats = roma.unitquat_to_rotmat(quats.roll(-1, dims=-1))
    else:
        rmats = cont_6d_to_rmat(init_rots)
    transls = init_ts
    transfms = rt_to_mat4(rmats, transls)
    center_tracks3d = torch.einsum(
        "bnij,bj->bni", transfms, F.pad(basis_centers, (0, 1), value=1.0)
    )[..., :3]
    vis_tracks_3d(server, center_tracks3d.cpu().numpy(), labels, name="se3_centers")


@torch.no_grad()
def vis_tracks_2d_video(
    path,
    imgs: np.ndarray,
    tracks_3d: np.ndarray,
    Ks: np.ndarray,
    w2cs: np.ndarray,
    occs=None,
    radius: int = 3,
):
    num_tracks = tracks_3d.shape[0]
    labels = np.linspace(0, 1, num_tracks)
    cmap = get_cmap("gist_rainbow")
    colors = cmap(labels)[:, :3]
    tracks_2d = (
        project_2d_tracks(tracks_3d.swapaxes(0, 1), Ks, w2cs).cpu().numpy()  # type: ignore
    )
    frames = np.asarray(
        draw_keypoints_video(imgs, tracks_2d, colors, occs, radius=radius)
    )
    iio.imwrite(path, frames, fps=15)


def vis_tracks_3d(
    server: ViserServer,
    vis_tracks: np.ndarray,
    vis_label: np.ndarray | None = None,
    name: str = "tracks",
):
    """
    :param vis_tracks (np.ndarray): (N, T, 3)
    :param vis_label (np.ndarray): (N)
    """
    cmap = get_cmap("gist_rainbow")
    if vis_label is None:
        vis_label = np.linspace(0, 1, len(vis_tracks))
    colors = cmap(np.asarray(vis_label))[:, :3]
    guru.info(f"{colors.shape=}, {vis_tracks.shape=}")
    N, T = vis_tracks.shape[:2]
    vis_tracks = np.asarray(vis_tracks)
    for i in range(N):
        server.scene.add_spline_catmull_rom(
            f"/{name}/{i}/spline", vis_tracks[i], color=colors[i], segments=T - 1
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/start",
            vis_tracks[i, [0]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/end",
            vis_tracks[i, [-1]],
            colors=colors[i : i + 1],
            point_size=0.05,
            point_shape="diamond",
        )


def sample_initial_bases_centers(
    mode: str, cano_t: int, tracks_3d: TrackObservations, num_bases: int
):
    """
    :param mode: "farthest" | "hdbscan" | "kmeans"
    :param tracks_3d: [G, T, 3]
    :param cano_t: canonical index
    :param num_bases: number of SE3 bases
    """
    assert mode in ["farthest", "hdbscan", "kmeans"]
    means_canonical = tracks_3d.xyz[:, cano_t].clone()
    # if mode == "farthest":
    #     vis_mask = tracks_3d.visibles[:, cano_t]
    #     sampled_centers, _ = sample_farthest_points(
    #         means_canonical[vis_mask][None],
    #         K=num_bases,
    #         random_start_point=True,
    #     )  # [1, num_bases, 3]
    #     dists2centers = torch.norm(means_canonical[:, None] - sampled_centers, dim=-1).T
    #     return sampled_centers, num_bases, dists2centers

    # linearly interpolate missing 3d points
    xyz = cp.asarray(tracks_3d.xyz)
    print(f"{xyz.shape=}")
    visibles = cp.asarray(tracks_3d.visibles)

    num_tracks = xyz.shape[0]
    xyz_interp = batched_interp_masked(xyz, visibles)

    # num_vis = 50
    # server = get_server(port=8890)
    # idcs = np.random.choice(num_tracks, num_vis)
    # labels = np.linspace(0, 1, num_vis)
    # vis_tracks_3d(server, tracks_3d.xyz[idcs].get(), labels, name="raw_tracks")
    # vis_tracks_3d(server, xyz_interp[idcs].get(), labels, name="interp_tracks")
    # import ipdb; ipdb.set_trace()

    velocities = xyz_interp[:, 1:] - xyz_interp[:, :-1]
    vel_dirs = (
        velocities / (cp.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-5)
    ).reshape((num_tracks, -1))

    # [num_bases, num_gaussians]
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    else:
        model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)
    model.fit(vel_dirs)
    labels = model.labels_
    num_bases = labels.max().item() + 1
    sampled_centers = torch.stack(
        [
            means_canonical[torch.tensor(labels == i)].median(dim=0).values
            for i in range(num_bases)
        ]
    )[None]
    print("number of {} clusters: ".format(mode), num_bases)
    return sampled_centers, num_bases, torch.tensor(labels)


def interp_masked(vals: cp.ndarray, mask: cp.ndarray, pad: int = 1) -> cp.ndarray:
    """
    hacky way to interpolate batched with cupy
    by concatenating the batches and pad with dummy values
    :param vals: [B, M, *]
    :param mask: [B, M]
    """
    assert mask.ndim == 2
    assert vals.shape[:2] == mask.shape

    B, M = mask.shape

    # get the first and last valid values for each track
    sh = vals.shape[2:]
    vals = vals.reshape((B, M, -1))
    D = vals.shape[-1]
    first_val_idcs = cp.argmax(mask, axis=-1)
    last_val_idcs = M - 1 - cp.argmax(cp.flip(mask, axis=-1), axis=-1)
    bidcs = cp.arange(B)

    v0 = vals[bidcs, first_val_idcs][:, None]
    v1 = vals[bidcs, last_val_idcs][:, None]
    m0 = mask[bidcs, first_val_idcs][:, None]
    m1 = mask[bidcs, last_val_idcs][:, None]
    if pad > 1:
        v0 = cp.tile(v0, [1, pad, 1])
        v1 = cp.tile(v1, [1, pad, 1])
        m0 = cp.tile(m0, [1, pad])
        m1 = cp.tile(m1, [1, pad])

    vals_pad = cp.concatenate([v0, vals, v1], axis=1)
    mask_pad = cp.concatenate([m0, mask, m1], axis=1)

    M_pad = vals_pad.shape[1]
    vals_flat = vals_pad.reshape((B * M_pad, -1))
    mask_flat = mask_pad.reshape((B * M_pad,))
    idcs = cp.where(mask_flat)[0]

    cx = cp.arange(B * M_pad)
    out = cp.zeros((B * M_pad, D), dtype=vals_flat.dtype)
    for d in range(D):
        out[:, d] = cp.interp(cx, idcs, vals_flat[idcs, d])

    out = out.reshape((B, M_pad, *sh))[:, pad:-pad]
    return out


def batched_interp_masked(
    vals: cp.ndarray, mask: cp.ndarray, batch_num: int = 4096, batch_time: int = 64
):
    assert mask.ndim == 2
    B, M = mask.shape
    out = cp.zeros_like(vals)
    for b in tqdm(range(0, B, batch_num), leave=False):
        for m in tqdm(range(0, M, batch_time), leave=False):
            x = interp_masked(
                vals[b : b + batch_num, m : m + batch_time],
                mask[b : b + batch_num, m : m + batch_time],
            )  # (batch_num, batch_time, *)
            out[b : b + batch_num, m : m + batch_time] = x
    return out
