import os
from dataclasses import asdict
from datetime import datetime

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from tqdm import tqdm

from flow3d.data import get_train_val_datasets
from flow3d.renderer import Renderer
from flow3d.trajectories import get_avg_w2c, get_lookat
from flow3d.vis.utils import (
    draw_keypoints_cv2,
    draw_tracks_2d,
    get_server,
    make_video_divisble,
)
from run_video import VideoConfig

torch.set_float32_matmul_precision("high")


def main(cfg: VideoConfig):
    train_dataset = get_train_val_datasets(cfg.data, load_val=False)[0]
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path)

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir=cfg.work_dir,
        port=None,
    )
    assert train_dataset.num_frames == renderer.num_frames

    guru.info(f"Rendering video from {renderer.global_step=}")

    K = train_dataset.get_Ks()[0].to(device)
    img_wh = train_dataset.get_img_wh()
    train_w2cs = train_dataset.get_w2cs().to(device)

    # select a keyframe
    i = len(train_dataset.keyframe_idcs) // 2
    tid = train_dataset.keyframe_idcs[i]
    tracks_3d = train_dataset.get_tracks_3d(1000)[0].to(device)  # (N, T, 3)
    avg_w2c = train_w2cs[tid]

    # move camera position back from the scene a bit
    scene_center = tracks_3d.reshape(-1, 3).mean(dim=0)
    lookat = scene_center - avg_w2c[:3, -1]
    avg_w2c[:3, -1] -= 0.2 * lookat

    # get the radius of the bounding sphere of training cameras
    train_c2ws = torch.linalg.inv(train_w2cs)
    rc_train_c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(avg_w2c), train_c2ws)
    rc_pos = rc_train_c2ws[:, :3, -1]
    rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 1.2
    print(f"{rads=}")
    lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
    up = torch.tensor([0.0, 0.0, 1.0], device=device)

    w2cs = cfg.trajectory.get_w2cs(
        ref_w2c=(
            avg_w2c
            if cfg.trajectory.ref_t < 0
            else train_w2cs[min(cfg.trajectory.ref_t, train_dataset.num_frames - 1)]
        ),
        lookat=lookat,
        up=up,
        focal_length=K[0, 0].item(),
        rads=rads,
        num_frames=len(train_w2cs),
        rots=0.5,
    )
    ts = cfg.time.get_ts(
        num_frames=len(train_w2cs),
        traj_frames=len(train_w2cs),
        device=device,
    )

    # w2cs = avg_w2c[None].repeat(num_frames, 1, 1)
    # ts = torch.arange(num_frames, device=device)
    assert len(w2cs) == len(ts)

    video = []
    grid = 16
    acc_thresh = 0.75
    window = 20
    # select gaussians with opacity > op_thresh
    # filter_mask = renderer.model.fg.get_opacities() > op_thresh

    # get tracks in world space
    train_i = 0
    with torch.inference_mode():
        render_outs = renderer.model.render(
            train_i,
            train_w2cs[train_i : train_i + 1],
            K[None],
            img_wh,
            target_ts=ts,
            return_color=True,
            fg_only=True,
            # filter_mask=filter_mask,
        )
    acc = render_outs["acc"][0].squeeze(-1)[::grid, ::grid]
    gt_mask = train_dataset.get_mask(0)[::grid, ::grid].to(device)  # (H, W)
    mask = (acc > acc_thresh) & (gt_mask > 0)

    # tracks in world space
    tracks_3d_map = render_outs["tracks_3d"][0][::grid, ::grid]  # (H, W, B, 3)
    mask = mask & ~(tracks_3d_map == 0).all(dim=(-1, -2))
    tracks_3d = tracks_3d_map[mask]  # (N, B, 3)
    print(f"{mask.sum()=} {tracks_3d.shape=}")

    tracks_2d = torch.einsum(
        "ij,bjk,nbk->nbi", K, w2cs[:, :3], F.pad(tracks_3d, (0, 1), value=1.0)
    )
    tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
    print(f"{tracks_2d.shape=}")

    # train_img = render_outs["img"][0]
    # train_img = (255 * train_img).cpu().numpy().astype(np.uint8)
    # kps = tracks_2d[:, 0].cpu().numpy()
    # server = get_server(8890)
    # import ipdb
    #
    # ipdb.set_trace()
    # server.scene.add_point_cloud(
    #     "points",
    #     tracks_3d_map[:, :, 0].cpu().numpy().reshape((-1, 3)),
    #     train_img[::grid, ::grid].reshape((-1, 3)),
    #     point_size=0.01,
    # )
    # train_img = draw_keypoints_cv2(train_img, kps)
    # iio.imwrite(f"{cfg.work_dir}/train_img.png", train_img)

    for i, (w2c, t) in enumerate(zip(tqdm(w2cs), ts)):
        i_min = max(0, i - window)
        if i - i_min < 1:
            continue
        with torch.inference_mode():
            img = renderer.model.render(int(t.item()), w2c[None], K[None], img_wh)[
                "img"
            ][0]
        out_img = draw_tracks_2d(img, tracks_2d[:, i_min:i])
        video.append(out_img)
    video = np.stack(video, 0)

    video_dir = f"{cfg.work_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(video_dir, exist_ok=True)
    iio.imwrite(f"{video_dir}/video.mp4", make_video_divisble(video), fps=cfg.fps)
    with open(f"{video_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)


if __name__ == "__main__":
    main(tyro.cli(VideoConfig))
