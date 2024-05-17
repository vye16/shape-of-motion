import warnings

import tyro
import os
from loguru import logger as guru
import pytorch_lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from viser import ViserServer

from flow3d.configs import SceneLRConfig, SceneLossesConfig, SceneOptimizerConfig
from flow3d.iphone_dataset import iPhoneDataset
from flow3d.init_model import InitMotionParams
from flow3d.init_utils import (
    init_bg_gaussians,
    init_fg_gaussians_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_tracks_3d,
)
from flow3d.scene_model import SceneModel
from flow3d.trainer import SceneModule
from flow3d.tensor_dataclass import StaticObservations, TrackObservations

warnings.filterwarnings("ignore", module="lightning")
L.pytorch.disable_possible_user_warnings()  # type: ignore

torch.set_float32_matmul_precision("high")


def vis_init_params(server, params: InitMotionParams, name="init_params"):
    num_vis = 50
    idcs = np.random.choice(params.num_gaussians, num_vis)
    labels = np.linspace(0, 1, num_vis)
    with torch.no_grad():
        pred_means = params.compute_means(
            torch.arange(params.num_frames, device=params.means.device)
        )
        vis_means = pred_means[idcs].detach().cpu().numpy()
    vis_tracks_3d(server, vis_means, labels, name=name)


def main(
    work_dir: str,
    data_dir: str,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int,
    port: int = 8890,
    batch_size: int = 8,
    num_dl_workers: int = 4,
    num_devices: int = 1,
):

    train_dataset = iPhoneDataset(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_dl_workers,
        collate_fn=iPhoneDataset.train_collate_fn,
    )
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")
    model = SceneModel(train_dataset.num_frames)
    lr_cfg = SceneLRConfig()
    loss_cfg = SceneLossesConfig()
    optim_cfg = SceneOptimizerConfig()
    scene_module = SceneModule(
        model, lr_cfg, loss_cfg, optim_cfg, work_dir=work_dir, port=port
    )

    # if checkpoint exists
    ckpt_path = os.path.join(work_dir, "last.ckpt")
    if not os.path.exists(ckpt_path):
        init_model_optim(train_dataset, model, num_fg, num_bg, num_motion_bases)
        guru.info(f"Saving initialization to {ckpt_path}")
        torch.save(scene_module.state_dict(), ckpt_path)
    else:
        guru.info(f"Loading checkpoint from {ckpt_path}")
        scene_module.load_state_dict(torch.load(ckpt_path))

    trainer = L.Trainer(accelerator="gpu", devices=num_devices)
    trainer.fit(scene_module, train_loader, ckpt_path="last")


def init_model_optim(
    train_dataset, model: SceneModel, num_fg: int, num_bg: int, num_motion_bases: int
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    assert tracks_3d.check_sizes()
    bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
    assert bg_points.check_sizes()

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())
    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")

    params, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t
    )
    fg_gaussians = init_fg_gaussians_from_tracks_3d(tracks_3d, cano_t)
    bg_gaussians = init_bg_gaussians(bg_points)

    # run initial optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = params.to(device)
    tracks_3d = tracks_3d.to(device)
    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)

    # server = ViserServer(port=8080)
    # vis_init_params(server, params)
    # run_initial_optim(params, tracks_3d, Ks, w2cs)
    # vis_init_params(server, params)

    fg_gaussians = fg_gaussians.to(device)
    fg_gaussians.means = params.means
    bg_gaussians = bg_gaussians.to(device)

    model.init_from_params(
        fg_gaussians,
        bg_gaussians,
        params.motion_coefs.data,
        params.motion_rots.data,
        params.motion_transls.data,
        Ks,
        w2cs,
    )
    return model


if __name__ == "__main__":
    tyro.cli(main)
