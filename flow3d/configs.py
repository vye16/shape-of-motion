from typing import Literal
from dataclasses import dataclass


@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4


@dataclass
class SceneLRConfig:
    fg: FGLRConfig = FGLRConfig()
    bg: BGLRConfig = BGLRConfig()
    motion_bases: MotionLRConfig = MotionLRConfig()


@dataclass
class LossesConfig:
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.1
    w_depth_grad: float = 1
    w_track: float = 2.0
    w_mask: float = 1.0
    w_smooth_bases: float = 0.1
    w_smooth_tracks: float = 0.0
    w_coef_similarity: float = 0.0
    w_pairwise_rigid: float = 0.0
    w_group_rigid: float = 0.0
    w_knn_rigid: float = 0.0
    knn_mode: Literal["dist", "coef"] = "dist"
    num_knn_samples: int = 64
    num_knn_neighbors: int = 16
    num_knn_masks: int = 4
    knn_beta: float = 2


@dataclass
class OptimizerConfig:
    max_steps: int = 5000
    ## Adaptive gaussian control
    warmup_steps: int = 500
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15
