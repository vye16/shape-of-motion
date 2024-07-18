import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated, Callable

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from loguru import logger as guru
from tqdm import tqdm

from flow3d.data import DavisDataConfig, get_train_val_datasets, iPhoneDataConfig
from flow3d.renderer import Renderer
from flow3d.trajectories import (
    get_arc_w2cs,
    get_avg_w2c,
    get_lemniscate_w2cs,
    get_lookat,
    get_spiral_w2cs,
    get_wander_w2cs,
)
from flow3d.vis.utils import make_video_divisble

torch.set_float32_matmul_precision("high")


@dataclass
class BaseTrajectoryConfig:
    num_frames: int = tyro.MISSING
    ref_t: int = -1
    _fn: tyro.conf.SuppressFixed[Callable] = tyro.MISSING

    def get_w2cs(self, **kwargs) -> torch.Tensor:
        cfg_kwargs = asdict(self)
        _fn = cfg_kwargs.pop("_fn")
        cfg_kwargs.update(kwargs)
        return _fn(**cfg_kwargs)


@dataclass
class ArcTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 120
    degree: float = 15.0
    _fn: tyro.conf.SuppressFixed[Callable] = get_arc_w2cs


@dataclass
class LemniscateTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 240
    degree: float = 15.0
    _fn: tyro.conf.SuppressFixed[Callable] = get_lemniscate_w2cs


@dataclass
class SpiralTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 240
    rads: float = 0.5
    zrate: float = 0.5
    rots: int = 2
    _fn: tyro.conf.SuppressFixed[Callable] = get_spiral_w2cs


@dataclass
class WanderTrajectoryConfig(BaseTrajectoryConfig):
    num_frames: int = 120
    _fn: tyro.conf.SuppressFixed[Callable] = get_wander_w2cs


@dataclass
class FixedTrajectoryConfig(BaseTrajectoryConfig):
    _fn: tyro.conf.SuppressFixed[Callable] = lambda ref_w2c, **_: ref_w2c[None]


@dataclass
class BaseTimeConfig:
    _fn: tyro.conf.SuppressFixed[Callable] = tyro.MISSING

    def get_ts(self, **kwargs) -> torch.Tensor:
        cfg_kwargs = asdict(self)
        _fn = cfg_kwargs.pop("_fn")
        return _fn(**kwargs, **cfg_kwargs)


@dataclass
class ReplayTimeConfig(BaseTimeConfig):
    _fn: tyro.conf.SuppressFixed[Callable] = (
        lambda num_frames, traj_frames, device, **_: F.pad(
            torch.arange(num_frames, device=device)[:traj_frames],
            (0, max(traj_frames - num_frames, 0)),
            value=num_frames - 1,
        )
    )


@dataclass
class FixedTimeConfig(BaseTimeConfig):
    t: int = 0
    _fn: tyro.conf.SuppressFixed[Callable] = (
        lambda t, num_frames, traj_frames, device, **_: torch.tensor(
            [min(t, num_frames - 1)], device=device
        ).expand(traj_frames)
    )


@dataclass
class VideoConfig:
    work_dir: str
    data: (
        Annotated[
            iPhoneDataConfig,
            tyro.conf.subcommand(
                name="iphone",
                default=iPhoneDataConfig(
                    data_dir=tyro.MISSING,
                    load_from_cache=True,
                    skip_load_imgs=True,
                ),
            ),
        ]
        | Annotated[
            DavisDataConfig,
            tyro.conf.subcommand(
                name="davis",
                default=DavisDataConfig(
                    seq_name=tyro.MISSING,
                    root_dir=tyro.MISSING,
                    load_from_cache=True,
                ),
            ),
        ]
    )
    trajectory: (
        Annotated[ArcTrajectoryConfig, tyro.conf.subcommand(name="arc")]
        | Annotated[LemniscateTrajectoryConfig, tyro.conf.subcommand(name="lemniscate")]
        | Annotated[SpiralTrajectoryConfig, tyro.conf.subcommand(name="spiral")]
        | Annotated[WanderTrajectoryConfig, tyro.conf.subcommand(name="wander")]
        | Annotated[FixedTrajectoryConfig, tyro.conf.subcommand(name="fixed")]
    )
    time: (
        Annotated[ReplayTimeConfig, tyro.conf.subcommand(name="replay")]
        | Annotated[FixedTimeConfig, tyro.conf.subcommand(name="fixed")]
    )
    fps: float = 15.0
    port: int = 8890


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

    train_w2cs = train_dataset.get_w2cs().to(device)
    avg_w2c = get_avg_w2c(train_w2cs)
    # avg_w2c = train_w2cs[0]
    train_c2ws = torch.linalg.inv(train_w2cs)
    lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    K = train_dataset.get_Ks()[0].to(device)
    img_wh = train_dataset.get_img_wh()

    # get the radius of the bounding sphere of training cameras
    rc_train_c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(avg_w2c), train_c2ws)
    rc_pos = rc_train_c2ws[:, :3, -1]
    rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 1.25

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
    )
    ts = cfg.time.get_ts(
        num_frames=renderer.num_frames,
        traj_frames=cfg.trajectory.num_frames,
        device=device,
    )

    import viser.transforms as vt
    from flow3d.vis.utils import get_server

    server = get_server(port=8890)
    for i, train_w2c in enumerate(train_w2cs):
        train_c2w = torch.linalg.inv(train_w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/train_camera/{i:03d}",
            np.pi / 4,
            1.0,
            0.02,
            (0, 0, 0),
            wxyz=vt.SO3.from_matrix(train_c2w[:3, :3]).wxyz,
            position=train_c2w[:3, -1],
        )
    for i, w2c in enumerate(w2cs):
        c2w = torch.linalg.inv(w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/camera/{i:03d}",
            np.pi / 4,
            1.0,
            0.02,
            (255, 0, 0),
            wxyz=vt.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, -1],
        )
        avg_c2w = torch.linalg.inv(avg_w2c).cpu().numpy()
        server.scene.add_camera_frustum(
            f"/ref_camera",
            np.pi / 4,
            1.0,
            0.02,
            (0, 0, 255),
            wxyz=vt.SO3.from_matrix(avg_c2w[:3, :3]).wxyz,
            position=avg_c2w[:3, -1],
        )
    import ipdb

    ipdb.set_trace()

    # num_frames = len(train_w2cs)
    # w2cs = train_w2cs[:1].repeat(num_frames, 1, 1)
    # ts = torch.arange(num_frames, device=device)
    # assert len(w2cs) == len(ts)

    video = []
    for w2c, t in zip(tqdm(w2cs), ts):
        with torch.inference_mode():
            img = renderer.model.render(int(t.item()), w2c[None], K[None], img_wh)[
                "img"
            ][0]
        img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        video.append(img)
    video = np.stack(video, 0)

    video_dir = f"{cfg.work_dir}/videos/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(video_dir, exist_ok=True)
    iio.imwrite(f"{video_dir}/video.mp4", make_video_divisble(video), fps=cfg.fps)
    with open(f"{video_dir}/cfg.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)


if __name__ == "__main__":
    main(tyro.cli(VideoConfig))
