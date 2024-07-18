from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE


def compute_psnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> float:
    """
    Args:
        preds (torch.Tensor): (..., 3) predicted images in [0, 1].
        targets (torch.Tensor): (..., 3) target images in [0, 1].
        masks (torch.Tensor | None): (...,) optional binary masks where the
            1-regions will be taken into account.

    Returns:
        psnr (float): Peak signal-to-noise ratio.
    """
    if masks is None:
        masks = torch.ones_like(preds[..., 0])
    return (
        -10.0
        * torch.log(
            F.mse_loss(
                preds * masks[..., None],
                targets * masks[..., None],
                reduction="sum",
            )
            / masks.sum().clamp(min=1.0)
            / 3.0
        )
        / np.log(10.0)
    ).item()


def compute_pose_errors(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float]:
    """
    Args:
        preds: (N, 4, 4) predicted camera poses.
        targets: (N, 4, 4) target camera poses.

    Returns:
        ate (float): Absolute trajectory error.
        rpe_t (float): Relative pose error in translation.
        rpe_r (float): Relative pose error in rotation (degree).
    """
    # Compute ATE.
    ate = torch.linalg.norm(preds[:, :3, -1] - targets[:, :3, -1], dim=-1).mean().item()
    # Compute RPE_t and RPE_r.
    # NOTE(hangg): It's important to use numpy here for the accuracy of RPE_r.
    # torch has numerical issues for acos when the value is close to 1.0, i.e.
    # RPE_r is supposed to be very small, and will result in artificially large
    # error.
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    pred_rels = np.linalg.inv(preds[:-1]) @ preds[1:]
    pred_rels = np.linalg.inv(preds[:-1]) @ preds[1:]
    target_rels = np.linalg.inv(targets[:-1]) @ targets[1:]
    error_rels = np.linalg.inv(target_rels) @ pred_rels
    traces = error_rels[:, :3, :3].trace(axis1=-2, axis2=-1)
    rpe_t = np.linalg.norm(error_rels[:, :3, -1], axis=-1).mean().item()
    rpe_r = (
        np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)).mean().item()
        / np.pi
        * 180.0
    )
    return ate, rpe_t, rpe_r


class mPSNR(PeakSignalNoiseRatio):
    sum_squared_error: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(self, **kwargs) -> None:
        super().__init__(
            data_range=1.0,
            base=10.0,
            dim=None,
            reduction="elementwise_mean",
            **kwargs,
        )
        self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (..., 3) float32 predicted images.
            targets (torch.Tensor): (..., 3) float32 target images.
            masks (torch.Tensor | None): (...,) optional binary masks where the
                1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        self.sum_squared_error.append(
            torch.sum(torch.pow((preds - targets) * masks[..., None], 2))
        )
        self.total.append(masks.sum().to(torch.int64) * 3)

    def compute(self) -> torch.Tensor:
        """Compute peak signal-to-noise ratio over state."""
        sum_squared_error = dim_zero_cat(self.sum_squared_error)
        total = dim_zero_cat(self.total)
        return -10.0 * torch.log(sum_squared_error / total).mean() / np.log(10.0)


class mSSIM(StructuralSimilarityIndexMeasure):
    similarity: list

    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=None,
            data_range=1.0,
            return_full_image=False,
            **kwargs,
        )
        assert isinstance(self.sigma, float)

    def __len__(self) -> int:
        return sum([s.shape[0] for s in self.similarity])

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional binary masks where
                the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])

        # Construct a 1D Gaussian blur filter.
        assert isinstance(self.kernel_size, int)
        hw = self.kernel_size // 2
        shift = (2 * hw - self.kernel_size + 1) / 2
        assert isinstance(self.sigma, float)
        f_i = (
            (torch.arange(self.kernel_size, device=preds.device) - hw + shift)
            / self.sigma
        ) ** 2
        filt = torch.exp(-0.5 * f_i)
        filt /= torch.sum(filt)

        # Blur in x and y (faster than the 2D convolution).
        def convolve2d(z, m, f):
            # z: (B, H, W, C), m: (B, H, W), f: (Hf, Wf).
            z = z.permute(0, 3, 1, 2)
            m = m[:, None]
            f = f[None, None].expand(z.shape[1], -1, -1, -1)
            z_ = torch.nn.functional.conv2d(
                z * m, f, padding="valid", groups=z.shape[1]
            )
            m_ = torch.nn.functional.conv2d(m, torch.ones_like(f[:1]), padding="valid")
            return torch.where(
                m_ != 0, z_ * torch.ones_like(f).sum() / (m_ * z.shape[1]), 0
            ).permute(0, 2, 3, 1), (m_ != 0)[:, 0].to(z.dtype)

        filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
        filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])
        filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

        mu0 = filt_fn(preds, masks)[0]
        mu1 = filt_fn(targets, masks)[0]
        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = filt_fn(preds**2, masks)[0] - mu00
        sigma11 = filt_fn(targets**2, masks)[0] - mu11
        sigma01 = filt_fn(preds * targets, masks)[0] - mu01

        # Clip the variances and covariances to valid values.
        # Variance must be non-negative:
        sigma00 = sigma00.clamp(min=0.0)
        sigma11 = sigma11.clamp(min=0.0)
        sigma01 = torch.sign(sigma01) * torch.minimum(
            torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
        )

        assert isinstance(self.data_range, float)
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
        denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
        ssim_map = numer / denom

        self.similarity.append(ssim_map.mean(dim=(1, 2, 3)))

    def compute(self) -> torch.Tensor:
        """Compute final SSIM metric."""
        return torch.cat(self.similarity).mean()


class mLPIPS(Metric):
    sum_scores: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that torchvision is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torchvision`."
            )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}."
            )
        self.net = _NoTrainLpips(net=net_type, spatial=True)

        self.add_state("sum_scores", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update internal states with lpips scores.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional float32 binary
                masks where the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        scores = self.net(
            (preds * masks[..., None]).permute(0, 3, 1, 2),
            (targets * masks[..., None]).permute(0, 3, 1, 2),
            normalize=True,
        )
        self.sum_scores.append((scores * masks[:, None]).sum())
        self.total.append(masks.sum().to(torch.int64))

    def compute(self) -> torch.Tensor:
        """Compute final perceptual similarity metric."""
        return (
            torch.tensor(self.sum_scores, device=self.device)
            / torch.tensor(self.total, device=self.device)
        ).mean()


class PCK(Metric):
    correct: list[torch.Tensor]
    total: list[int]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, threshold: float):
        """Update internal states with PCK scores.

        Args:
            preds (torch.Tensor): (N, 2) predicted 2D keypoints.
            targets (torch.Tensor): (N, 2) targets 2D keypoints.
            threshold (float): PCK threshold.
        """

        self.correct.append(
            (torch.linalg.norm(preds - targets, dim=-1) < threshold).sum()
        )
        self.total.append(preds.shape[0])

    def compute(self) -> torch.Tensor:
        """Compute PCK over state."""
        return (
            torch.tensor(self.correct, device=self.device)
            / torch.clamp(torch.tensor(self.total, device=self.device), min=1e-8)
        ).mean()
