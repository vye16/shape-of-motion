from dataclasses import asdict
import functools
import time
from typing import cast
from nerfview.utils import CameraState, with_view_lock
import numpy as np
from loguru import logger as guru
from pytorch_msssim import SSIM
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from flow3d.loss_utils import (
    compute_gradient_loss,
    masked_l1_loss,
    compute_z_acc_loss,
    compute_se3_smoothness_loss,
)
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.configs import SceneLRConfig, LossesConfig, OptimizerConfig


class Trainer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        # Logging.
        work_dir: str,
        port: int | None = None,
        hang_on_complete: bool = True,
        log_every: int = 10,
        checkpoint_every: int | None = 200,
        validate_every: int | None = 500,
        validate_video_every: int | None = 1000,
        validate_viewer_assets_every: int | None = 100,
    ):
        self.device = device
        self.validate_video_every = validate_video_every
        self.validate_viewer_assets_every = validate_viewer_assets_every

        self.model = model
        self.num_frames = model.num_frames

        self.lr_cfg = lr_cfg
        self.losses_cfg = losses_cfg
        self.optim_cfg = optim_cfg

        self.reset_opacity_every = (
            self.optim_cfg.reset_opacity_every_n_controls * self.optim_cfg.control_every
        )
        self.optimizers, self.scheduler = self.configure_optimizers()
        self.writer = SummaryWriter(log_dir=work_dir)
        self.global_step = 0

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS()
        self.pck_metric = PCK()
        self.bg_psnr_metric = mPSNR()
        self.fg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.fg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS()
        self.fg_lpips_metric = mLPIPS()

        self._xys_grad_norm_acc = None
        self._visible_num_steps = None
        self._max_normalized_radii = None

    def load_state_dict(self, state_dict: dict):
        model_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                name = k.split("model.")[-1]
                model_dict[name] = v

        self.model.load_state_dict(model_dict)

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        img = self.model.render(camera_state.extras["t"], w2c[None], K[None], img_wh)[
            "img"
        ][0]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def train_step(self, batch):
        loss, num_rays_per_step, num_rays_per_sec = self.compute_losses(batch)
        loss.backward()

        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in self.scheduler.values():
            sched.step()

        self.run_control_steps()
        return loss.item()

    def compute_losses(self, batch):
        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # (B, H, W).
        depths = batch["depths"]
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]

        _tic = time.time()
        # (B, G, 3).
        means, quats = self.model.compute_poses_all(ts)  # (G, B, 3), (G, B, 4)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        # [(N, G, 3), ...].
        target_ts_vec = torch.cat(target_ts)
        # (B * N, G, 3).
        target_means, _ = self.model.compute_poses_all(target_ts_vec)
        target_means = target_means.transpose(0, 1)
        target_mean_list = target_means.split(N)
        num_frames = self.model.num_frames

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            bg_color = torch.ones(3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                target_ts=target_ts[i],
                target_w2cs=target_w2cs[i],
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                target_means=target_mean_list[i].transpose(0, 1),
                return_depth=True,
                return_mask=self.model.has_bg,
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)
            self._batched_xys.append(self.model._current_xys)
            self._batched_radii.append(self.model._current_radii)
            self._batched_img_wh.append(self.model._current_img_wh)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([i[key] for i in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.stack(bg_colors)

        # Compute losses.
        # (B * N).
        frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
        if not self.model.has_bg:
            imgs = (
                imgs * masks[..., None]
                + (1.0 - masks[..., None]) * bg_colors[:, None, None]
            )
        else:
            imgs = (
                imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )
        # (P_all, 2).
        tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0)
        # (P_all,)
        visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
        # (P_all,)
        confidences = torch.cat([x.reshape(-1) for x in target_confidences], dim=0)

        # RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        if self.model.has_bg:
            rendered_imgs = (
                rendered_imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )
        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * 1.0

        # Mask loss.
        if not self.model.has_bg:
            mask_loss = F.mse_loss(rendered_all["acc"], masks[..., None])  # type: ignore
        else:
            mask_loss = F.mse_loss(
                rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore
            ) + masked_l1_loss(
                rendered_all["mask"],
                masks[..., None],
                quantile=0.98,  # type: ignore
            )
        loss += mask_loss * self.losses_cfg.w_mask

        # (B * N, H * W, 3).
        pred_tracks_3d = (
            rendered_all["tracks_3d"].permute(0, 3, 1, 2, 4).reshape(-1, H * W, 3)  # type: ignore
        )
        pred_tracks_2d = torch.einsum(
            "bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d
        )
        # (B * N, H * W, 1).
        mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)
        # (B * N, H * W, 2).
        pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth

        # (B * N).
        w_interval = torch.exp(-2 * frame_intervals / num_frames)
        # w_track_loss = min(1, (self.max_steps - self.global_step) / 6000)
        track_weights = confidences[..., None] * w_interval

        # (B, H, W).
        masks_flatten = torch.zeros_like(masks)
        for i in range(B):
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
            query_pixels = query_tracks_2d[i].to(torch.int64)
            masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0
        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
        )

        track_2d_loss = masked_l1_loss(
            pred_tracks_2d[masks_flatten][visibles],
            tracks_2d[visibles],
            mask=track_weights[visibles],
            quantile=0.98,
        ) / max(H, W)
        loss += track_2d_loss * self.losses_cfg.w_track

        depth_masks = (
            masks[..., None] if not self.model.has_bg else valid_masks[..., None]
        )

        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks,
            quantile=0.98,
        )
        # depth_loss = cauchy_loss_with_uncertainty(
        #     pred_disp.squeeze(-1),
        #     tgt_disp.squeeze(-1),
        #     depth_masks.squeeze(-1),
        #     self.depth_uncertainty_activation(self.depth_uncertainties)[ts],
        #     bias=1e-3,
        # )
        loss += depth_loss * self.losses_cfg.w_depth_reg

        # mapped depth loss (using cached depth with EMA)
        #  mapped_depth_loss = 0.0
        mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
        mapped_depth_loss = masked_l1_loss(
            1 / (mapped_depth[masks_flatten][visibles] + 1e-5),
            1 / (mapped_depth_gt[visibles, None] + 1e-5),
            track_weights[visibles],
        )

        loss += mapped_depth_loss * self.losses_cfg.w_depth_const

        #  depth_gradient_loss = 0.0
        depth_gradient_loss = compute_gradient_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks > 0.5,
            quantile=0.95,
        )
        # depth_gradient_loss = compute_gradient_loss(
        #     pred_disps,
        #     ref_disps,
        #     mask=depth_masks.squeeze(-1) > 0.5,
        #     c=depth_uncertainty.detach(),
        #     mode="l1",
        #     bias=1e-3,
        # )
        loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

        # bases should be smooth.
        small_accel_loss = compute_se3_smoothness_loss(
            self.model.motion_bases.params["rots"],
            self.model.motion_bases.params["transls"],
        )
        loss += small_accel_loss * self.losses_cfg.w_smooth_bases

        # tracks should be smooth
        ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        transfms_nbs = self.model.compute_transforms(ts_neighbors)  # (G, 3n, 3, 4)
        means_fg_nbs = torch.einsum(
            "pnij,pj->pni",
            transfms_nbs,
            F.pad(self.model.fg.params["means"], (0, 1), value=1.0),
        )
        means_fg_nbs = means_fg_nbs.reshape(
            means_fg_nbs.shape[0], 3, -1, 3
        )  # [G, 3, n, 3]
        if self.losses_cfg.w_smooth_tracks > 0:
            small_accel_loss_tracks = (
                (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
                .norm(dim=-1)
                .mean()
            )
            loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks * 0.5

        # Constrain the std of scales.
        # TODO: do we want to penalize before or after exp?
        loss += 0.01 * torch.var(self.model.fg.params["scales"], dim=-1).mean()
        if self.model.bg is not None:
            loss += 0.01 * torch.var(self.model.bg.params["scales"], dim=-1).mean()

        # # sparsity loss
        # loss += 0.01 * self.opacity_activation(self.opacities).abs().mean()

        # Acceleration along ray direction should be small.
        z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs)
        loss += z_accel_loss

        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/mapped_depth_loss": mapped_depth_loss.item(),
            "train/track_2d_loss": track_2d_loss.item(),
            "train/small_accel_loss": small_accel_loss.item(),
            "train/z_acc_loss": z_accel_loss.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
        }

        # Compute metrics.
        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, masks if not self.model.has_bg else valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr
            if self.model.has_bg:
                bg_psnr = self.bg_psnr_metric(rendered_imgs, imgs, 1.0 - masks)
                fg_psnr = self.fg_psnr_metric(rendered_imgs, imgs, masks)
                self.bg_psnr_metric.reset()
                self.fg_psnr_metric.reset()
                stats["train/bg_psnr"] = bg_psnr
                stats["train/fg_psnr"] = fg_psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        # Log stats.
        self.log_dict( stats)
        self.global_step += 1

        return loss, num_rays_per_step, num_rays_per_sec

    def log_dict(self, stats: dict):
        for k, v in stats.items():
            self.writer.add_scalar(k, v, self.global_step)

    # def compute_z_acc_loss(self, ts: torch.Tensor):
    #     """
    #     :param ts: [B]
    #     :param w2cs: [B, 4, 4]
    #     """
    #     num_frames = self.model.num_frames
    #     ts = torch.clamp(ts, min=1, max=num_frames - 2)
    #     ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
    #     means, _ = self.model.compute_poses_fg(ts_neighbors)  # [G, 3n, 3]
    #     means = means.reshape(means.shape[0], 3, -1, 3)  # [G, 3, n, 3]
    #     camera_center_t = torch.linalg.inv(self.model.w2cs[ts])[:, :3, 3]
    #     ray_dir = F.normalize(means[:, 1] - camera_center_t, p=2.0, dim=-1)  # [G, n, 3]
    #     # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, n, 3]
    #     # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
    #     acc_loss = (((means[:, 1] - means[:, 0]) * ray_dir).sum(dim=-1) ** 2).mean() + (
    #         ((means[:, 2] - means[:, 1]) * ray_dir).sum(dim=-1) ** 2
    #     ).mean()
    #     return acc_loss
    #
    # def compute_motion_bases_smoothness_loss(
    #     self, weight_rot: float = 1.0, weight_transl: float = 2.0
    # ):
    #     small_acc_loss = 0.0
    #     rots = self.model.motion_rots
    #     small_accel_loss_q = 2 * rots[:, 1:-1] - rots[:, :-2] - rots[:, 2:]
    #     small_acc_loss += small_accel_loss_q.norm(dim=-1).mean() * weight_rot
    #
    #     transls = self.model.motion_transls
    #     small_accel_loss_t = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
    #     small_acc_loss += small_accel_loss_t.norm(dim=-1).mean() * weight_transl
    #     return small_acc_loss

    def run_control_steps(self):
        global_step = self.global_step
        # Adaptive gaussian control.
        cfg = self.optim_cfg
        num_frames = self.model.num_frames
        self._prepare_control_step()
        if (
            global_step > cfg.warmup_steps
            and global_step % cfg.control_every == 0
            and global_step < cfg.stop_control_steps
        ):
            if (
                global_step < cfg.stop_densify_steps
                and global_step % self.reset_opacity_every > num_frames
            ):
                self._densify_control_step(global_step)
            if global_step % self.reset_opacity_every > min(3 * num_frames, 1000):
                self._cull_control_step(global_step)
            if global_step % self.reset_opacity_every == 0:
                self._reset_opacity_control_step()

            # Reset stats after every control.
            self._xys_grad_norm_acc = None
            self._visible_num_steps = None
            self._max_normalized_radii = None

    @with_view_lock
    @torch.no_grad()
    def _prepare_control_step(self):
        # Prepare for adaptive gaussian control based on the current stats.
        assert (
            self.model._current_radii is not None
            and self.model._current_xys is not None
            and self._batched_xys is not None
            and self._batched_radii is not None
        )
        batch_size = len(self._batched_xys)
        for _current_xys, _current_radii, _current_img_wh in zip(
            self._batched_xys, self._batched_radii, self._batched_img_wh
        ):
            visible_masks = (_current_radii > 0).flatten()
            xys_grad_norm = torch.linalg.norm(_current_xys.grad, dim=-1) * batch_size
            if self._xys_grad_norm_acc is None:
                self._xys_grad_norm_acc = xys_grad_norm
                self._visible_num_steps = torch.ones_like(
                    xys_grad_norm, dtype=torch.int64
                )
            else:
                assert self._visible_num_steps is not None
                self._xys_grad_norm_acc[visible_masks] += xys_grad_norm[visible_masks]
                self._visible_num_steps[visible_masks] += 1
            if self._max_normalized_radii is None:
                self._max_normalized_radii = torch.zeros_like(
                    self.model._current_radii, dtype=torch.float32
                )
            assert (
                self._max_normalized_radii is not None and _current_img_wh is not None
            )
            self._max_normalized_radii[visible_masks] = torch.maximum(
                self._max_normalized_radii[visible_masks],
                _current_radii[visible_masks] / max(_current_img_wh),
            )

    @with_view_lock
    @torch.no_grad()
    def _densify_control_step(self, global_step):
        assert (
            self._xys_grad_norm_acc is not None
            and self._visible_num_steps is not None
            and self._max_normalized_radii is not None
            and self._batched_img_wh is not None
        )
        cfg = self.optim_cfg
        xys_grad_norm = (
            (self._xys_grad_norm_acc / self._visible_num_steps)
            * 0.5
            * max(max(self._batched_img_wh))
        )
        is_grad_too_high = xys_grad_norm > cfg.densify_xys_grad_threshold
        # Split gaussians.
        scales = self.model.get_scales_all()
        is_scale_too_big = scales.max(dim=-1)[0] > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self._max_normalized_radii > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        should_dup = is_grad_too_high & ~is_scale_too_big

        num_fg = self.model.num_fg_gaussians
        should_fg_split = should_split[:num_fg]
        num_fg_splits = int(should_fg_split.sum().item())
        should_fg_dup = should_dup[:num_fg]
        num_fg_dups = int(should_fg_dup.sum().item())

        should_bg_split = should_split[num_fg:]
        num_bg_splits = int(should_bg_split.sum().item())
        should_bg_dup = should_dup[num_fg:]
        num_bg_dups = int(should_bg_dup.sum().item())

        fg_param_map = self.model.fg.densify_params(should_fg_split, should_fg_dup)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            dup_in_optim(
                optimizer,
                [new_params],
                should_fg_split,
                num_fg_splits * 2 + num_fg_dups,
            )

        if self.model.bg is not None:
            bg_param_map = self.model.bg.densify_params(should_bg_split, should_bg_dup)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                dup_in_optim(
                    optimizer,
                    [new_params],
                    should_bg_split,
                    num_bg_splits * 2 + num_bg_dups,
                )

        device = scales.device
        # Update this since _cull_control_step might use it.
        self._max_normalized_radii = torch.cat(
            [
                self._max_normalized_radii[:num_fg][~should_fg_split],
                torch.zeros(num_fg_splits * 2, device=device),
                torch.zeros(num_fg_dups, device=device),
                self._max_normalized_radii[num_fg:][~should_bg_split],
                torch.zeros(num_bg_splits * 2, device=device),
                torch.zeros(num_bg_dups, device=device),
            ],
            dim=0,
        )
        guru.info(
            f"Splitted {should_split.sum().item()} gaussians, "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _cull_control_step(self, global_step):
        # Cull gaussians.
        cfg = self.optim_cfg
        opacities = self.model.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = (opacities < cfg.cull_opacity_threshold)[:, 0]
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )
        num_fg = self.model.num_fg_gaussians
        cull_scale_threshold[num_fg:] *= self.model.bg_scene_scale
        if global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.max(dim=-1)[0] > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                assert self._max_normalized_radii is not None
                is_radius_too_big = (
                    self._max_normalized_radii > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:]

        fg_param_map = self.model.fg.cull_params(should_fg_cull)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_fg_cull)

        if self.model.bg is not None:
            bg_param_map = self.model.bg.cull_params(should_bg_cull)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                remove_from_optim(optimizer, [new_params], should_bg_cull)

        guru.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _reset_opacity_control_step(self):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        fg_params = self.model.reset_opacities(new_val)
        # Modify optimizer states by new assignment.
        for param_name, new_params in fg_params.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            reset_in_optim(optimizer, [new_params])

    def configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        for name, params in self.model.named_parameters():
            mod, _, field = name.split(".")
            lr = lr_dict[mod][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name:
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()
