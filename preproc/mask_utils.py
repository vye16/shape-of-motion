import glob

import numpy as np
from loguru import logger as guru
from segment_anything import SamPredictor, sam_model_registry
from tracker.base_tracker import BaseTracker


def init_sam_model(checkpoint_dir: str, sam_model_type: str, device) -> SamPredictor:
    checkpoints = glob.glob(f"{checkpoint_dir}/*{sam_model_type}*.pth")
    if len(checkpoints) == 0:
        raise ValueError(
            f"No checkpoints found for model type {sam_model_type} in {checkpoint_dir}"
        )
    checkpoints = sorted(checkpoints)
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoints[-1])
    sam.to(device=device)
    guru.info(f"loaded model checkpoint {checkpoints[-1]}")
    return SamPredictor(sam)


def init_tracker(checkpoint_dir, device) -> BaseTracker:
    checkpoints = glob.glob(f"{checkpoint_dir}/*XMem*.pth")
    if len(checkpoints) == 0:
        raise ValueError(f"No XMem checkpoints found in {checkpoint_dir}")
    checkpoints = sorted(checkpoints)
    return BaseTracker(checkpoints[-1], device)


def track_masks(
    tracker: BaseTracker,
    imgs_np: np.ndarray | list,
    cano_mask: np.ndarray,
    cano_t: int,
):
    """
    :param imgs_np: (T, H, W, 3)
    :param masks: (H, W)
    """
    T = len(imgs_np)
    cano_mask = cano_mask > 0.5

    # forward from canonical_id
    masks_forward = []
    for t in range(int(cano_t), T):
        frame = imgs_np[t]
        if t == cano_t:
            mask = tracker.track(frame, cano_mask)
        else:
            mask = tracker.track(frame)
        masks_forward.append(mask)
    tracker.clear_memory()

    # backward from canonical_id
    masks_backward = []
    for t in range(int(cano_t), -1, -1):
        frame = imgs_np[t]
        if t == cano_t:
            mask = tracker.track(frame, cano_mask)
        else:
            mask = tracker.track(frame)
        masks_backward.append(mask)
    tracker.clear_memory()

    masks_all = masks_backward[::-1] + masks_forward[1:]
    return masks_all
