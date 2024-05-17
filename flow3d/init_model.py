import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.transforms import cont_6d_to_rmat


class InitMotionParams(nn.Module):
    def __init__(self, motion_rots, motion_transls, motion_coefs, means):
        super().__init__()
        self.num_frames = motion_rots.shape[1]
        self.num_gaussians = means.shape[0]
        self.motion_rots = nn.Parameter(motion_rots)
        self.motion_transls = nn.Parameter(motion_transls)
        self.motion_coefs = nn.Parameter(motion_coefs)
        self.means = nn.Parameter(means)

    def compute_transforms(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts (torch.Tensor): (B,).

        Returns:
            torch.Tensor: (G, B, 3, 4).
        """
        motion_transls = self.motion_transls[:, ts].swapaxes(0, 1)
        motion_rots = self.motion_rots[:, ts]  # (K, B, 6)
        coefs = F.softmax(self.motion_coefs, dim=-1)
        transls = torch.einsum("pk,nki->pni", coefs, motion_transls)
        rots = torch.einsum("pk,kni->pni", coefs, motion_rots)  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)

    def compute_means(self, ts: torch.Tensor):
        transfms = self.compute_transforms(ts)
        return torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(self.means, (0, 1), value=1.0),
        )

    def get_motion_coefs(self):
        return F.softmax(self.motion_coefs, dim=-1)

    def compute_motion_bases_smoothness_loss(
        self, weight_rot: float = 1.0, weight_transl: float = 2.0
    ):
        small_acc_loss = 0.0
        small_accel_loss_r = (
            2 * self.motion_rots[:, 1:-1]
            - self.motion_rots[:, :-2]
            - self.motion_rots[:, 2:]
        )
        small_acc_loss += small_accel_loss_r.norm(dim=-1).mean() * weight_rot

        small_accel_loss_t = (
            2 * self.motion_transls[:, 1:-1]
            - self.motion_transls[:, :-2]
            - self.motion_transls[:, 2:]
        )
        small_acc_loss += small_accel_loss_t.norm(dim=-1).mean() * weight_transl
        return small_acc_loss

    def compute_z_acc_loss(self, ts: torch.Tensor, w2cs: torch.Tensor):
        ts = torch.clamp(ts, min=1, max=self.num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))  # i (3B,)
        means = self.compute_means(ts_neighbors)  # (G, 3B, 3)
        means = means.reshape(means.shape[0], 3, -1, 3)  # [G, 3, n, 3]
        camera_center_t = torch.linalg.inv(w2cs[ts])[:, :3, 3]  # (B, 3)
        ray_dir = F.normalize(means[:, 1] - camera_center_t, p=2.0, dim=-1)  # [G, n, 3]
        # acc = 2 * means[:, 1] - means[:, 0] - means[:, 2]  # [G, n, 3]
        # acc_loss = (acc * ray_dir).sum(dim=-1).abs().mean()
        acc_loss = (((means[:, 1] - means[:, 0]) * ray_dir).sum(dim=-1) ** 2).mean() + (
            ((means[:, 2] - means[:, 1]) * ray_dir).sum(dim=-1) ** 2
        ).mean()
        return acc_loss
