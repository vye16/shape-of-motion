import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rendering import rasterization_2dgs
from torch import Tensor

from flow3d.params import GaussianParams, MotionBases


class SceneModel(nn.Module):
    def __init__(
        self,
        Ks: Tensor,
        w2cs: Tensor,
        fg_params: GaussianParams,
        motion_bases: MotionBases,
        bg_params: GaussianParams | None = None,
    ):
        super().__init__()
        self.num_frames = motion_bases.num_frames
        self.fg = fg_params
        self.motion_bases = motion_bases
        self.bg = bg_params
        scene_scale = 1.0 if bg_params is None else bg_params.scene_scale
        self.register_buffer("bg_scene_scale", torch.tensor(scene_scale))
        self.register_buffer("Ks", Ks)
        self.register_buffer("w2cs", w2cs)

        self._current_xys = None
        self._current_radii = None
        self._current_img_wh = None

    @property
    def num_gaussians(self) -> int:
        return self.num_bg_gaussians + self.num_fg_gaussians

    @property
    def num_bg_gaussians(self) -> int:
        return self.bg.num_gaussians if self.bg is not None else 0

    @property
    def num_fg_gaussians(self) -> int:
        return self.fg.num_gaussians

    @property
    def num_motion_bases(self) -> int:
        return self.motion_bases.num_bases

    @property
    def has_bg(self) -> bool:
        return self.bg is not None

    def compute_poses_bg(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            means: (G, B, 3)
            quats: (G, B, 4)
        """
        assert self.bg is not None
        return self.bg.params["means"], self.bg.get_quats()

    def compute_transforms(self, ts: torch.Tensor) -> torch.Tensor:
        coefs = self.fg.get_coefs()  # (G, K)
        transfms = self.motion_bases.compute_transforms(ts, coefs)  # (G, B, 3, 4)
        return transfms

    def compute_poses_fg(self, ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :returns means: (G, B, 3), quats: (G, B, 4)
        """
        transfms = self.compute_transforms(ts)  # (G, B, 3, 4)
        means = self.fg.params["means"]  # (G, 3)
        means = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(means, (0, 1), value=1.0),
        )
        quats = roma.quat_xyzw_to_wxyz(
            (
                roma.quat_product(
                    roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                    roma.quat_wxyz_to_xyzw(self.fg.get_quats()[:, None]),
                )
            )
        )
        quats = F.normalize(quats, p=2, dim=-1)
        return means, quats

    def compute_poses_all(self, ts) -> tuple[torch.Tensor, torch.Tensor]:
        means, quats = self.compute_poses_fg(ts)
        if self.has_bg:
            bg_means, bg_quats = self.compute_poses_bg()
            means = torch.cat(
                [means, bg_means[:, None].expand(-1, len(ts), -1)], dim=0
            ).contiguous()
            quats = torch.cat(
                [quats, bg_quats[:, None].expand(-1, len(ts), -1)], dim=0
            ).contiguous()
        return means, quats

    def get_colors_all(self) -> torch.Tensor:
        colors = self.fg.get_colors()
        if self.bg is not None:
            colors = torch.cat([colors, self.bg.get_colors()], dim=0).contiguous()
        return colors

    def get_scales_all(self) -> torch.Tensor:
        scales = self.fg.get_scales()
        if self.bg is not None:
            scales = torch.cat([scales, self.bg.get_scales()], dim=0).contiguous()
        return scales

    def get_opacities_all(self) -> torch.Tensor:
        """
        :returns colors: (G, 3), scales: (G, 3), opacities: (G, 1)
        """
        opacities = self.fg.get_opacities()
        if self.bg is not None:
            opacities = torch.cat(
                [opacities, self.bg.get_opacities()], dim=0
            ).contiguous()
        return opacities

    @staticmethod
    def init_from_state_dict(state_dict, prefix=""):
        fg = GaussianParams.init_from_state_dict(
            state_dict, prefix=f"{prefix}fg.params."
        )
        bg = None
        if any("bg." in k for k in state_dict):
            bg = GaussianParams.init_from_state_dict(
                state_dict, prefix=f"{prefix}bg.params."
            )
        motion_bases = MotionBases.init_from_state_dict(
            state_dict, prefix=f"{prefix}motion_bases.params."
        )
        Ks = state_dict[f"{prefix}Ks"]
        w2cs = state_dict[f"{prefix}w2cs"]
        return SceneModel(Ks, w2cs, fg, motion_bases, bg)

    def render(
        self,
        # A single time instance for view rendering.
        t: int,
        w2cs: torch.Tensor,  # (C, 4, 4)
        Ks: torch.Tensor,  # (C, 3, 3)
        img_wh: tuple[int, int],
        # Multiple time instances for track rendering: (B,).
        target_ts: torch.Tensor | None = None,  # (B)
        target_w2cs: torch.Tensor | None = None,  # (B, 4, 4)
        fg_only_tracks: bool = False,
        bg_color: torch.Tensor | float = 1.0,
        colors_override: torch.Tensor | None = None,
        means: torch.Tensor | None = None,
        quats: torch.Tensor | None = None,
        target_means: torch.Tensor | None = None,
        return_depth: bool = False,
        return_mask: bool = False,
    ) -> dict:
        device = w2cs.device
        C = w2cs.shape[0]

        W, H = img_wh

        if means is None or quats is None:
            means, quats = self.compute_poses_all(torch.tensor([t], device=device))
            means = means[:, 0]
            quats = quats[:, 0]

        if colors_override is None:
            colors_override = self.get_colors_all()
        D = colors_override.shape[-1]

        scales = self.get_scales_all()
        opacities = self.get_opacities_all()

        if isinstance(bg_color, float):
            bg_color = torch.full((C, D), bg_color, device=device)
        assert isinstance(bg_color, torch.Tensor)

        mode = "RGB"
        ds_expected = {"img": D}

        return_mask &= self.has_bg
        if return_mask:
            mask_values = torch.zeros((self.num_gaussians, 1), device=device)
            mask_values[: self.num_fg_gaussians] = 1.0
            colors_override = torch.cat([colors_override, mask_values], dim=-1)
            bg_color = torch.cat([bg_color, torch.zeros(C, 1, device=device)], dim=-1)
            ds_expected["mask"] = 1

        B = 0
        if target_ts is not None:
            if target_means is None:
                target_means, _ = self.compute_poses_all(target_ts)  # [G, B, 3]
            assert target_w2cs is not None
            B = target_ts.shape[0]
            target_means = torch.einsum(
                "bij,pbj->pbi",
                target_w2cs[:, :3],
                F.pad(target_means, (0, 1), value=1.0),
            )
            track_3d_vals = target_means.flatten(-2)  # (G, B * 3)
            d_track = track_3d_vals.shape[-1]
            colors_override = torch.cat([colors_override, track_3d_vals], dim=-1)
            bg_color = torch.cat(
                [bg_color, torch.zeros(C, track_3d_vals.shape[-1], device=device)],
                dim=-1,
            )
            ds_expected["tracks_3d"] = d_track

        assert colors_override.shape[-1] == sum(ds_expected.values())
        assert bg_color.shape[-1] == sum(ds_expected.values())

        if return_depth:
            mode = "RGB+ED"
            ds_expected["depth"] = 1

        (render_colors, alphas, _, _), info = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_override,
            backgrounds=bg_color,
            viewmats=w2cs,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=W,
            height=H,
            packed=False,
            render_mode=mode,
        )

        # Populate the current data for adaptive gaussian control.
        if self.training and info["means2d"].requires_grad:
            self._current_xys = info["means2d"]
            self._current_radii = info["radii"]
            self._current_img_wh = img_wh
            # We want to be able to access to xys' gradients later in a
            # torch.no_grad context.
            self._current_xys.retain_grad()

        assert render_colors.shape[-1] == sum(ds_expected.values())
        outputs = torch.split(render_colors, list(ds_expected.values()), dim=-1)
        out_dict = {}
        for i, (name, dim) in enumerate(ds_expected.items()):
            x = outputs[i]
            assert x.shape[-1] == dim, f"{x.shape[-1]=} != {dim=}"
            if name == "tracks_3d":
                x = x.reshape(C, H, W, B, 3)
            out_dict[name] = x
        out_dict["acc"] = alphas
        return out_dict
