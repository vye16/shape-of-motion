from dataclasses import asdict
import os
import os.path as osp
import functools
import time
from tqdm import tqdm
from typing import cast
from nerfview import CameraState, Viewer
import numpy as np
import imageio as iio
from loguru import logger as guru
from pytorch_msssim import SSIM
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.configs import SceneLRConfig, LossesConfig, OptimizerConfig
from flow3d.data_utils import to_device, normalize_coords


class Validator:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        val_img_dataloader: DataLoader | None,
        val_kpt_dataloader: DataLoader,
        save_dir: str,
    ):
        self.model = model
        self.device = device
        self.val_img_dataloader = val_img_dataloader
        self.val_kpt_dataloader = val_kpt_dataloader
        self.save_dir = save_dir
        self.has_bg = self.model.has_bg

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS().to(device)
        self.fg_psnr_metric = mPSNR()
        self.fg_ssim_metric = mSSIM()
        self.fg_lpips_metric = mLPIPS().to(device)
        self.bg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS().to(device)
        self.pck_metric = PCK()

    def reset_metrics(self):
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.lpips_metric.reset()
        self.fg_psnr_metric.reset()
        self.fg_ssim_metric.reset()
        self.fg_lpips_metric.reset()
        self.bg_psnr_metric.reset()
        self.bg_ssim_metric.reset()
        self.bg_lpips_metric.reset()
        self.pck_metric.reset()

    @torch.no_grad()
    def validate(self):
        self.reset_metrics()
        metric_imgs = self.validate_imgs()
        metric_kpts = self.validate_keypoints()
        return {**metric_imgs, **metric_kpts}

    @torch.no_grad()
    def validate_imgs(self):
        guru.info("rendering validation images...")
        if self.val_img_dataloader is None:
            return

        for batch in tqdm(self.val_img_dataloader, desc="render val images"):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0]
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]
            # (1, H, W).
            valid_mask = batch.get(
                "valid_masks", torch.ones_like(batch["imgs"][..., 0])
            )
            # (1, H, W).
            fg_mask = batch["masks"]

            # (H, W).
            covisible_mask = batch.get(
                "covisible_masks",
                torch.ones_like(fg_mask)[None],
            )
            W, H = img_wh = img[0].shape[-2::-1]
            rendered = self.model.render(t, w2c, K, img_wh, return_depth=True)

            # Compute metrics.
            valid_mask *= covisible_mask
            fg_valid_mask = fg_mask * valid_mask
            bg_valid_mask = (1 - fg_mask) * valid_mask
            main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)

                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

            # Dump results.
            results_dir = osp.join(self.save_dir, "results", "rgb")
            os.makedirs(results_dir, exist_ok=True)
            iio.imwrite(
                osp.join(results_dir, f"{frame_name}.png"),
                (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
            )

        return {
            "val/psnr": self.psnr_metric.compute(),
            "val/ssim": self.ssim_metric.compute(),
            "val/lpips": self.lpips_metric.compute(),
            "val/fg_psnr": self.fg_psnr_metric.compute(),
            "val/fg_ssim": self.fg_ssim_metric.compute(),
            "val/fg_lpips": self.fg_lpips_metric.compute(),
            "val/bg_psnr": self.bg_psnr_metric.compute(),
            "val/bg_ssim": self.bg_ssim_metric.compute(),
            "val/bg_lpips": self.bg_lpips_metric.compute(),
        }

    @torch.no_grad()
    def validate_keypoints(self):
        pred_keypoints_3d_all = []
        time_ids = self.val_kpt_dataloader.dataset.time_ids.tolist()
        h, w = self.val_kpt_dataloader.dataset.dataset.imgs.shape[1:3]
        pred_train_depths = np.zeros((len(time_ids), h, w))

        for batch in tqdm(self.val_kpt_dataloader, desc="render val keypoints"):
            batch = to_device(batch, self.device)
            # (2,).
            ts = batch["ts"][0]
            # (2, 4, 4).
            w2cs = batch["w2cs"][0]
            # (2, 3, 3).
            Ks = batch["Ks"][0]
            # (2, H, W, 3).
            imgs = batch["imgs"][0]
            # (2, P, 3).
            keypoints = batch["keypoints"][0]
            # (P,)
            keypoint_masks = (keypoints[..., -1] > 0.5).all(dim=0)
            src_keypoints, target_keypoints = keypoints[:, keypoint_masks, :2]
            W, H = img_wh = imgs.shape[-2:0:-1]
            rendered = self.model.render(
                ts[0].item(),
                w2cs[:1],
                Ks[:1],
                img_wh,
                target_ts=ts[1:],
                target_w2cs=w2cs[1:],
                return_depth=True,
            )
            pred_tracks_3d = rendered["tracks_3d"][0, ..., 0, :]
            pred_tracks_2d = torch.einsum("ij,hwj->hwi", Ks[1], pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., -1:], min=1e-6
            )
            pred_keypoints = F.grid_sample(
                pred_tracks_2d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]

            # Compute metrics.
            self.pck_metric.update(pred_keypoints, target_keypoints, max(img_wh) * 0.05)

            padded_keypoints_3d = torch.zeros_like(keypoints[0])
            pred_keypoints_3d = F.grid_sample(
                pred_tracks_3d[None].permute(0, 3, 1, 2),
                normalize_coords(src_keypoints, H, W)[None, None],
                align_corners=True,
            ).permute(0, 2, 3, 1)[0, 0]
            # Transform 3D keypoints back to world space.
            pred_keypoints_3d = torch.einsum(
                "ij,pj->pi",
                torch.linalg.inv(w2cs[1])[:3],
                F.pad(pred_keypoints_3d, (0, 1), value=1.0),
            )
            padded_keypoints_3d[keypoint_masks] = pred_keypoints_3d
            # Cache predicted keypoints.
            pred_keypoints_3d_all.append(padded_keypoints_3d.cpu().numpy())
            pred_train_depths[time_ids.index(ts[0].item())] = (
                rendered["depth"][0, ..., 0].cpu().numpy()
            )

        # Dump unified results.
        all_Ks = self.val_kpt_dataloader.dataset.dataset.Ks
        all_w2cs = self.val_kpt_dataloader.dataset.dataset.w2cs

        keypoint_result_dict = {
            "Ks": all_Ks[time_ids].cpu().numpy(),
            "w2cs": all_w2cs[time_ids].cpu().numpy(),
            "pred_keypoints_3d": np.stack(pred_keypoints_3d_all, 0),
            "pred_train_depths": pred_train_depths,
        }

        results_dir = osp.join(self.save_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        np.savez(
            osp.join(results_dir, "keypoints.npz"),
            **keypoint_result_dict,
        )
        guru.info(
            f"Dumped keypoint results to {results_dir=} {keypoint_result_dict['pred_keypoints_3d'].shape=}"
        )

        return {"val/pck": self.pck_metric.compute()}
