from dataclasses import asdict, dataclass
import warnings
from tqdm import tqdm

import tyro
import os
from loguru import logger as guru
import numpy as np
import torch
from torch.utils.data import DataLoader

from flow3d.configs import SceneLRConfig, LossesConfig, OptimizerConfig
from flow3d.iphone_dataset import DataConfig, iPhoneDataset
from flow3d.params import GaussianParams, MotionBases
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
)
from flow3d.scene_model import SceneModel
from flow3d.trainer import Trainer
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.vis_utils import get_server

torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:
    work_dir: str
    data_cfg: DataConfig
    num_fg: int = 40_000
    num_bg: int = 100_000
    num_motion_bases: int = 10
    num_epochs: int = 500
    port: int = 8890
    batch_size: int = 8
    num_dl_workers: int = 4


def main(cfg: TrainConfig):

    train_dataset = iPhoneDataset(**asdict(cfg.data_cfg))
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_cfg = SceneLRConfig()
    loss_cfg = LossesConfig()
    optim_cfg = OptimizerConfig()

    # if checkpoint exists
    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    initialize_and_checkpoint_model(cfg, train_dataset, device, ckpt_path)

    trainer = Trainer.init_from_checkpoint(
        ckpt_path,
        device,
        lr_cfg,
        loss_cfg,
        optim_cfg,
        work_dir=cfg.work_dir,
        port=cfg.port,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dl_workers,
        collate_fn=iPhoneDataset.train_collate_fn,
    )
    guru.info(f"Starting training from {trainer.global_step=}")
    for _ in (pbar := tqdm(range(cfg.num_epochs))):
        for batch in train_loader:
            batch = to_device(batch, device)
            loss = trainer.train_step(batch)
            pbar.set_description(f"Loss: {loss}")


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [to_device(v, device) for v in batch]
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_dataset: iPhoneDataset,
    device: torch.device,
    ckpt_path: str,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return

    fg_params, motion_bases, bg_params, tracks_3d = init_model_from_tracks(
        train_dataset, cfg.num_fg, cfg.num_bg, cfg.num_motion_bases
    )
    # run initial optimization
    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)
    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs)
    server = get_server(port=8890)
    vis_init_params(server, fg_params, motion_bases)
    model = SceneModel(Ks, w2cs, fg_params, motion_bases, bg_params)

    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)


def init_model_from_tracks(
    train_dataset, num_fg: int, num_bg: int, num_motion_bases: int
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    assert tracks_3d.check_sizes()

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())
    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
        assert bg_points.check_sizes()
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, bg_params, tracks_3d


if __name__ == "__main__":
    tyro.cli(main)
