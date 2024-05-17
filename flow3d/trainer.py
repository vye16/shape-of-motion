import functools
import time
from typing import Literal, cast
from nerfview.utils import CameraState, with_view_lock
import numpy as np
from loguru import logger as guru
from pytorch_msssim import SSIM
import torch
import torch.nn.functional as F
from loss_utils import compute_gradient_loss, masked_l1_loss
from pytorch_lightning import LightningModule
from metrics import PCK, mLPIPS, mPSNR, mSSIM
from scene_model import SceneModel
from configs import SceneLRConfig, SceneLossesConfig, SceneOptimizerConfig


class Trainer(LightningModule):
    def __init__(
        self,
        model: SceneModel,
        lr_cfg: SceneLRConfig,
        losses_cfg: SceneLossesConfig,
        optim_cfg: SceneOptimizerConfig,
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
        super().__init__(
            work_dir=work_dir,
            port=port,
            hang_on_complete=hang_on_complete,
            max_steps=optim_cfg.max_steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            validate_every=validate_every,
        )
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

    def load_state_dict(self, state_dict: dict, **kwargs):  # type: ignore
        model_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                name = k.split("model.")[-1]
                model_dict[name] = v

        self.model.load_state_dict(model_dict)
        super().load_state_dict(state_dict, strict=False)

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
        img = self.model.render(camera_state.extras["t"], w2c, K, img_wh)["img"]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def training_step(self, batch, *_, **__):  # type: ignore
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
                w2cs[i],
                Ks[i],
                img_wh,
                target_ts=target_ts[i],
                target_w2cs=target_w2cs[i],
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                target_means=target_mean_list[i].transpose(0, 1),
                return_alpha=True,
                return_depth=True,
                return_mask=self.model.has_bg,
                return_coefs=self.losses_cfg.w_coef_similarity > 0,
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
                torch.stack([i[key] for i in rendered_all])
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

        # Motion should be smooth.
        small_accel_loss = self.compute_motion_bases_smoothness_loss()
        loss += small_accel_loss * self.losses_cfg.w_smooth_bases

        if self.losses_cfg.w_smooth_tracks > 0:
            ts = torch.clamp(ts, min=1, max=num_frames - 2)
            ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
            means_neighbors, quats_neighbors = self.model.compute_poses_fg(
                ts_neighbors
            )  # [G, 3n, 3]
            means_neighbors = means_neighbors.reshape(
                means_neighbors.shape[0], 3, -1, 3
            )  # [G, 3, n, 3]
            small_accel_loss_tracks = (
                (
                    2 * means_neighbors[:, 1:-1]
                    - means_neighbors[:, :-2]
                    - means_neighbors[:, 2:]
                )
                .norm(dim=-1)
                .mean()
            )
            loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks * 0.5

        # Constrain the std of scales.
        # TODO: do we want to penalize before or after exp?
        loss += 0.01 * torch.var(self.model.scales, dim=-1).mean()
        if self.model.has_bg:
            loss += 0.01 * torch.var(self.model.bg_scales, dim=-1).mean()

        # # sparsity loss
        # loss += 0.01 * self.opacity_activation(self.opacities).abs().mean()

        num_fg = self.model.num_fg_gaussians
        num_bg = self.model.num_bg_gaussians

        # Acceleration along ray direction should be small.
        z_accel_loss = self.compute_z_acc_loss(ts)
        loss += z_accel_loss

        # Prepare stats for logging.
        stats = {
            "train/loss": loss,
            "train/rgb_loss": rgb_loss,
            "train/mask_loss": mask_loss,
            "train/depth_loss": depth_loss,
            "train/depth_graident_loss": depth_gradient_loss,
            "train/mapped_depth_loss": mapped_depth_loss,
            "train/track_2d_loss": track_2d_loss,
            "train/small_accel_loss": small_accel_loss,
            "train/z_acc_loss": z_accel_loss,
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_bg_gaussians": num_bg,
            "train/num_fg_gaussians": num_fg,
        }

        # Compute metrics.
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
        self.log_dict(stats)

        return {
            "loss": loss,
            "num_rays_per_step": num_rays_per_step,
            "num_rays_per_sec": num_rays_per_sec,
        }

    def compute_z_acc_loss(self, ts: torch.Tensor):
        """
        :param ts: [B]
        :param w2cs: [B, 4, 4]
        """
        num_frames = self.model.num_frames
        ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        means, _ = self.model.compute_poses_fg(ts_neighbors)  # [G, 3n, 3]
        means = means.reshape(means.shape[0], 3, -1, 3)  # [G, 3, n, 3]
        camera_center_t = torch.linalg.inv(self.model.w2cs[ts])[:, :3, 3]
        ray_dir = F.normalize(means[:, 1] - camera_center_t, p=2.0, dim=-1)  # [G, n, 3]
        # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, n, 3]
        # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
        acc_loss = (((means[:, 1] - means[:, 0]) * ray_dir).sum(dim=-1) ** 2).mean() + (
            ((means[:, 2] - means[:, 1]) * ray_dir).sum(dim=-1) ** 2
        ).mean()
        return acc_loss

    def compute_motion_bases_smoothness_loss(
        self, weight_rot: float = 1.0, weight_transl: float = 2.0
    ):
        small_acc_loss = 0.0
        rots = self.model.motion_rots
        small_accel_loss_q = 2 * rots[:, 1:-1] - rots[:, :-2] - rots[:, 2:]
        small_acc_loss += small_accel_loss_q.norm(dim=-1).mean() * weight_rot

        transls = self.model.motion_transls
        small_accel_loss_t = 2 * transls[:, 1:-1] - transls[:, :-2] - transls[:, 2:]
        small_acc_loss += small_accel_loss_t.norm(dim=-1).mean() * weight_transl
        return small_acc_loss

    def on_train_batch_end(self, *_, **__):
        super().on_train_batch_end(*_, **__)
        # Adaptive gaussian control.
        cfg = self.optim_cfg
        num_frames = self.model.num_frames
        self._prepare_control_step()
        if (
            self.global_step > cfg.warmup_steps
            and self.global_step % cfg.control_every == 0
            and self.global_step < cfg.stop_control_steps
        ):
            if (
                self.global_step < cfg.stop_densify_steps
                and self.global_step % self.reset_opacity_every > num_frames
            ):
                self._densify_control_step()
            if self.global_step % self.reset_opacity_every > min(3 * num_frames, 1000):
                self._cull_control_step()
            if self.global_step % self.reset_opacity_every == 0:
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
    def _densify_control_step(self):
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
        if self.global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self._max_normalized_radii > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        num_fg = self.model.num_fg_gaussians
        should_fg_split = should_split[:num_fg]
        num_fg_splits = int(should_fg_split.sum().item())
        should_bg_split = should_split[num_fg:]
        num_bg_splits = int(should_bg_split.sum().item())

        should_dup = is_grad_too_high & ~is_scale_too_big
        should_fg_dup = should_dup[:num_fg]
        num_fg_dups = int(should_fg_dup.sum().item())
        should_bg_dup = should_dup[num_fg:]
        num_bg_dups = int(should_bg_dup.sum().item())

        fg_param_map = self.model.densify_fg_params(should_fg_split, should_fg_dup)
        for param_name, new_params in fg_param_map.items():
            idx = self._param_name_to_group_idx[param_name]
            self.dup_in_optim(
                idx, new_params, should_fg_split, num_fg_splits * 2 + num_fg_dups
            )

        bg_param_map = self.model.densify_bg_params(should_bg_split, should_bg_dup)
        for param_name, new_params in bg_param_map.items():
            idx = self._param_name_to_group_idx[param_name]
            self.dup_in_optim(
                idx, new_params, should_bg_split, num_bg_splits * 2 + num_bg_dups
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

    def dup_in_optim(self, param_idx: int, new_params, should_dup, num_dups: int):
        optimizer = self.optimizer
        old_params = optimizer.param_groups[param_idx]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        param_state["exp_avg"] = torch.cat(
            [
                param_state["exp_avg"][~should_dup],
                param_state["exp_avg"].new_zeros(
                    num_dups, *param_state["exp_avg"].shape[1:]
                ),
            ],
            dim=0,
        )
        param_state["exp_avg_sq"] = torch.cat(
            [
                param_state["exp_avg_sq"][~should_dup],
                param_state["exp_avg_sq"].new_zeros(
                    num_dups, *param_state["exp_avg_sq"].shape[1:]
                ),
            ],
            dim=0,
        )
        del optimizer.state[old_params]
        optimizer.state[new_params] = param_state
        optimizer.param_groups[param_idx]["params"] = [new_params]
        del old_params

    @torch.no_grad()
    def _cull_control_step(self):
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
        if self.global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.max(dim=-1)[0] > cull_scale_threshold
            if self.global_step < cfg.stop_control_by_screen_steps:
                assert self._max_normalized_radii is not None
                is_radius_too_big = (
                    self._max_normalized_radii > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:]

        fg_param_map = self.model.cull_fg_params(should_fg_cull)
        bg_param_map = self.model.cull_bg_params(should_bg_cull)
        for param_name, new_params in fg_param_map.items():
            idx = self._param_name_to_group_idx[param_name]
            self.remove_from_optim(idx, new_params, should_fg_cull)

        for param_name, new_params in bg_param_map.items():
            idx = self._param_name_to_group_idx[param_name]
            self.remove_from_optim(idx, new_params, should_bg_cull)

        guru.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    def remove_from_optim(self, param_idx: int, new_params, _should_cull):
        optimizer = self.optimizer
        old_params = optimizer.param_groups[param_idx]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        param_state["exp_avg"] = param_state["exp_avg"][~_should_cull]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[new_params] = param_state
        optimizer.param_groups[param_idx]["params"] = [new_params]
        del old_params

    @torch.no_grad()
    def _reset_opacity_control_step(self):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        updated_params = self.model.reset_opacities(new_val)
        # Modify optimizer states by new assignment.
        for param_name, new_params in updated_params.items():
            idx = self._param_name_to_group_idx[param_name]
            self.reset_in_optim(idx, new_params)

    def reset_in_optim(self, param_idx: int, new_params):
        optimizer = self.optimizer
        old_params = optimizer.param_groups[param_idx]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
        del optimizer.state[old_params]
        optimizer.state[new_params] = param_state
        optimizer.param_groups[param_idx]["params"] = [new_params]
        del old_params

    def configure_optimizers(self):
        param_groups = [
            {
                "params": getattr(self.model, name),
                "lr": getattr(self.lr_cfg, name),
            }
            for name in self.model.param_names
        ]
        self._param_name_to_group_idx = {
            name: i for i, name in enumerate(self.model.param_names)
        }

        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_lambdas = []
        for name in self.model.param_names:
            lr = getattr(self.lr_cfg, name)
            if name == "scales":
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0
            lr_lambdas.append(functools.partial(fnc, lr_init=lr))

        optimizer = torch.optim.Adam(param_groups, eps=1e-15)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas),
            "interval": "step",
        }
        self.optimizer = optimizer
        self.scheduler = scheduler
        return [optimizer], [scheduler]
