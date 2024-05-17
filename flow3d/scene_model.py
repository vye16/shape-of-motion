import math
import roma
import torch
import torch.nn as nn
from loguru import logger as guru
from torch import Tensor
import torch.nn.functional as F

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from flow3d.transforms import cont_6d_to_rmat
from flow3d.tensor_dataclass import GaussianParams


class SceneModel(nn.Module):
    def __init__(self, num_frames: int):
        super().__init__()
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)
        self.num_frames = num_frames

        self._current_xys = None
        self._current_radii = None
        self._current_img_wh = None

    def init_from_params(
        self,
        init_fg_gaussians: GaussianParams,
        init_bg_gaussians: GaussianParams,
        init_motion_coefs: Tensor,
        init_motion_rots: Tensor,
        init_motion_transls: Tensor,
        Ks: Tensor,
        w2cs: Tensor,
    ):
        assert init_motion_rots.shape[-1] in [4, 6]
        assert self.num_frames == init_motion_rots.shape[1]
        assert self.num_frames == len(w2cs)
        assert self.num_frames == len(Ks)
        num_bases = init_motion_coefs.shape[-1]
        assert num_bases == init_motion_rots.shape[0]
        assert num_bases == init_motion_transls.shape[0]

        self.register_parameter(f"means", nn.Parameter(init_fg_gaussians.means))
        self.register_parameter(f"quats", nn.Parameter(init_fg_gaussians.quats))
        self.register_parameter(f"colors", nn.Parameter(init_fg_gaussians.colors))
        self.register_parameter(f"scales", nn.Parameter(init_fg_gaussians.scales))
        self.register_parameter(f"opacities", nn.Parameter(init_fg_gaussians.opacities))

        self.register_parameter(f"bg_means", nn.Parameter(init_bg_gaussians.means))
        self.register_parameter(f"bg_quats", nn.Parameter(init_bg_gaussians.quats))
        self.register_parameter(f"bg_colors", nn.Parameter(init_bg_gaussians.colors))
        self.register_parameter(f"bg_scales", nn.Parameter(init_bg_gaussians.scales))
        self.register_parameter(
            f"bg_opacities", nn.Parameter(init_bg_gaussians.opacities)
        )

        self.register_parameter(
            "motion_coefs", nn.Parameter(init_motion_coefs)
        )  # (G, K)
        self.register_parameter(
            "motion_rots", nn.Parameter(init_motion_rots)
        )  # (K, T, 6)
        self.register_parameter(
            "motion_transls", nn.Parameter(init_motion_transls)
        )  # (K, T, 3)
        self.register_buffer(
            "bg_scene_scale", torch.tensor(init_bg_gaussians.scene_scale)
        )
        self.register_buffer("Ks", Ks)
        self.register_buffer("w2cs", w2cs)

    def init_from_state_dict(self, state_dict, strict: bool = True):
        for name, param in state_dict.items():
            if name in self.param_names:
                self.register_parameter(name, nn.Parameter(param))
            elif name in self.buffer_names:
                self.register_buffer(name, param)
            else:
                msg = f"Unknown parameter: {name}"
                if strict:
                    raise ValueError(msg)
                else:
                    guru.warning(msg)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs) -> None:  # type: ignore
        if not hasattr(self, "means"):
            self.init_from_state_dict(state_dict, strict=strict)
        else:
            super().load_state_dict(state_dict, strict=strict, **kwargs)

    @property
    def buffer_names(self):
        return ["Ks", "w2cs", "bg_scene_scale"]

    @property
    def fg_param_names(self):
        return ["means", "quats", "colors", "scales", "opacities", "motion_coefs"]

    @property
    def bg_param_names(self):
        return ["bg_means", "bg_quats", "bg_colors", "bg_scales", "bg_opacities"]

    @property
    def motion_bases_param_names(self):
        return ["motion_rots", "motion_transls"]

    @property
    def param_names(self):
        return self.fg_param_names + self.bg_param_names + self.motion_bases_param_names

    @property
    def num_gaussians(self) -> int:
        return self.num_bg_gaussians + self.num_fg_gaussians

    @property
    def num_bg_gaussians(self) -> int:
        return self.bg_colors.shape[0]

    @property
    def num_fg_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def num_motion_bases(self) -> int:
        return self.motion_rots.shape[0]

    @property
    def has_bg(self) -> bool:
        return self.num_bg_gaussians > 0

    def compute_transforms(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts (torch.Tensor): (B,).

        Returns:
            torch.Tensor: (G, B, 3, 4).
        """
        motion_transls = self.motion_transls[:, ts]  # (K, B, 3)
        motion_rots = self.motion_rots[:, ts]  # (K, B, 6)
        motion_coefs = self.motion_coef_activation(self.motion_coefs)
        transls = torch.einsum("pk,kni->pni", motion_coefs, motion_transls)
        rots = torch.einsum("pk,kni->pni", motion_coefs, motion_rots)  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)

    def compute_poses_bg(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            means: (G, B, 3)
            quats: (G, B, 4)
        """
        assert self.has_bg
        return self.bg_means, self.quat_activation(self.bg_quats)

    def compute_poses_fg(self, ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :returns means: (G, B, 3), quats: (G, B, 4)
        """
        transfms = self.compute_transforms(ts)  # (G, B, 3, 4)
        means = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(self.means, (0, 1), value=1.0),
        )
        quats = roma.quat_xyzw_to_wxyz(
            (
                roma.quat_product(
                    roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                    roma.quat_wxyz_to_xyzw(self.quat_activation(self.quats[:, None])),
                )
            )
        )
        return means, self.quat_activation(quats)

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
        colors = self.colors
        if self.has_bg:
            colors = torch.cat([self.colors, self.bg_colors], dim=0).contiguous()
        return self.color_activation(colors)

    def get_scales_all(self) -> torch.Tensor:
        scales = self.scales
        if self.has_bg:
            scales = torch.cat([self.scales, self.bg_scales], dim=0).contiguous()
        return self.scale_activation(scales)

    def get_opacities_all(self) -> torch.Tensor:
        """
        :returns colors: (G, 3), scales: (G, 3), opacities: (G, 1)
        """
        opacities = self.opacities
        if self.has_bg:
            opacities = torch.cat(
                [self.opacities, self.bg_opacities], dim=0
            ).contiguous()
        return self.opacity_activation(opacities)

    def render(
        self,
        # A single time instance for view rendering.
        t: int,
        w2c: torch.Tensor,
        K: torch.Tensor,
        img_wh: tuple[int, int],
        # Multiple time instances for track rendering: (B,).
        target_ts: torch.Tensor | None = None,
        target_w2cs: torch.Tensor | None = None,
        fg_only_tracks: bool = False,
        bg_color: torch.Tensor | float = 1.0,
        colors_override: torch.Tensor | None = None,
        means: torch.Tensor | None = None,
        quats: torch.Tensor | None = None,
        target_means: torch.Tensor | None = None,
        return_alpha: bool = False,
        return_depth: bool = False,
        return_mask: bool = False,
        return_coefs: bool = False,
    ) -> dict:
        device = self.colors.device
        if isinstance(bg_color, float):
            bg_color = torch.full((3,), bg_color, device=device)
        assert isinstance(bg_color, torch.Tensor)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        W, H = img_wh

        BLOCK_SIZE = 16

        if means is None or quats is None:
            means, quats = self.compute_poses_all(torch.tensor([t], device=device))
            means = means[:, 0]
            quats = quats[:, 0]

        colors = self.get_colors_all()
        scales = self.get_scales_all()
        opacities = self.get_opacities_all()

        xys, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
            means,
            scales,
            1.0,
            quats,
            w2c[:3],
            fx.item(),
            fy.item(),
            cx.item(),
            cy.item(),
            H,
            W,
            BLOCK_SIZE,
        )

        # Populate the current data for adaptive gaussian control.
        if self.training and xys.requires_grad:
            self._current_xys = xys
            self._current_radii = radii
            self._current_img_wh = img_wh
            # We want to be able to access to xys' gradients later in a
            # torch.no_grad context.
            self._current_xys.retain_grad()

        rast_out = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors if colors_override is None else colors_override,
            opacities,
            H,
            W,
            BLOCK_SIZE,
            bg_color,
            return_alpha=return_alpha,
        )
        if return_alpha:
            img = rast_out[0]
            acc = rast_out[1][..., None]
        else:
            img = rast_out
            acc = None

        depth = None
        if return_depth:
            depth = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                depths[..., None],
                opacities,
                H,
                W,
                BLOCK_SIZE,
                bg_color.new_zeros(1),
            )[..., :1]

        mask = None
        if return_mask and self.has_bg:  # render foreground mask
            mask_values = torch.zeros(self.num_gaussians, device=device)
            mask_values[: self.num_fg_gaussians] = 1.0
            mask = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                mask_values[..., None],
                opacities,
                H,
                W,
                BLOCK_SIZE,
                bg_color.new_zeros(1),
            )[..., :1]

        coef_img = None
        if return_coefs:
            coefs = self.motion_coef_activation(self.motion_coefs).contiguous()
            coef_img = rasterize_gaussians(
                xys[: self.num_fg_gaussians],
                depths[: self.num_fg_gaussians],
                radii[: self.num_fg_gaussians],
                conics[: self.num_fg_gaussians],
                num_tiles_hit[: self.num_fg_gaussians],
                coefs,
                opacities[: self.num_fg_gaussians],
                H,
                W,
                BLOCK_SIZE,
                torch.zeros(coefs.shape[-1], device=device),
                return_alpha=False,
            )

        tracks_3d = None
        if target_ts is not None:
            if target_means is None:
                target_means, _ = self.compute_poses_all(target_ts)  # [G, B, 3]
            assert target_w2cs is not None
            target_means = torch.einsum(
                "bij,bpj->bpi",
                target_w2cs[:, :3],
                F.pad(target_means.transpose(0, 1), (0, 1), value=1.0),
            ).transpose(0, 1)
            num_tgt_render = (
                self.num_fg_gaussians if fg_only_tracks else self.num_gaussians
            )
            rast_values = target_means.flatten(-2)[:num_tgt_render]
            tracks_3d = rasterize_gaussians(
                xys[:num_tgt_render],
                depths[:num_tgt_render],
                radii[:num_tgt_render],
                conics[:num_tgt_render],
                num_tiles_hit[:num_tgt_render],
                rast_values,
                opacities[:num_tgt_render],
                H,
                W,
                BLOCK_SIZE,
                rast_values.new_zeros(rast_values.shape[-1]),
            )
            tracks_3d = tracks_3d.reshape(H, W, target_ts.shape[0], 3)

        return {
            # (H, W, 3).
            "img": img,
            # (H, W, 1) or None, depending on only_img.
            "depth": depth,
            # (H, W, 1) or None, depending on only_img.
            "acc": acc,
            # (H, W, D) or None, depending on render_coefs.
            "coef": coef_img,
            # (H, W, 1) or None, depending on only_img and model_bg.
            "mask": mask,
            # (H, W, B, 3) or None, depending on target_ts.
            "tracks_3d": tracks_3d,
        }

    def densify_fg_params(self, should_fg_split, should_fg_dup):
        """
        separate functions for densifying fg and bg params
        """
        updated_params = {}
        for name in self.fg_param_names:
            x = getattr(self, name)
            x_dup = x[should_fg_dup]
            x_split = x[should_fg_split].repeat(2, 1)
            if name == "scales":
                x_split -= math.log(1.6)
            x_new = torch.cat([x[~should_fg_split], x_dup, x_split], dim=0)
            updated_params[name] = nn.Parameter(x_new)
            self.register_parameter(name, updated_params[name])
        return updated_params

    def densify_bg_params(self, should_bg_split, should_bg_dup):
        """
        separate functions for densifying fg and bg params
        """
        updated_params = {}
        for name in self.bg_param_names:
            x = getattr(self, name)
            x_dup = x[should_bg_dup]
            x_split = x[should_bg_split].repeat(2, 1)
            x_new = torch.cat([x[~should_bg_split], x_dup, x_split], dim=0)
            updated_params[name] = nn.Parameter(x_new)
            self.register_parameter(name, updated_params[name])
        return updated_params

    def cull_fg_params(self, should_fg_cull):
        """
        separate functions for culling fg and bg params
        """
        updated_params = {}
        for name in self.fg_param_names:
            x = getattr(self, name)
            x_new = x[~should_fg_cull]
            updated_params[name] = nn.Parameter(x_new)
            self.register_parameter(name, updated_params[name])
        return updated_params

    def cull_bg_params(self, should_bg_cull):
        """
        separate functions for culling fg and bg params
        """
        updated_params = {}
        for name in self.bg_param_names:
            x = getattr(self, name)
            x_new = x[~should_bg_cull]
            updated_params[name] = nn.Parameter(x_new)
            self.register_parameter(name, updated_params[name])
        return updated_params

    def reset_opacities(self, new_val):
        """
        separate functions for culling fg and bg params
        """
        self.opacities.data.fill_(new_val)
        updated_params = {"opacities": self.opacities}
        if self.has_bg:
            self.bg_opacities.data.fill_(new_val)
            updated_params["bg_opacities"] = self.bg_opacities
        return updated_params
