import glob
import os
import os.path as osp

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from tracker.inference.inference_core import InferenceCore
from tracker.model.network import XMem
from tracker.util.mask_mapper import MaskMapper
from tracker.util.range_transform import im_normalization


class BaseTracker(object):
    def __init__(self, xmem_checkpoint, device) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        #  with open("tracker/config/config.yaml", "r") as stream:
        with open(
            osp.join(osp.dirname(__file__), "config", "config.yaml"), "r"
        ) as stream:
            config = yaml.safe_load(stream)
        # initialise XMem
        network = XMem(config, xmem_checkpoint).to(device).eval()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.device = device

        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input:
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:  # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None

        # prepare inputs
        frame_tensor = self.im_transform(frame).to(self.device)
        # track one frame
        probs, _ = self.tracker.step(frame_tensor, mask, labels)  # logits 2 (bg fg) H W
        # # refine
        # if first_frame_annotation is None:
        #     out_mask = self.sam_refinement(frame, logits[1], ti)

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)

        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        return final_mask

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()


@torch.no_grad()
def sam_refinement(sam_model, frame, logits):
    """
    refine segmentation results with mask prompt
    :param frame (H, W, 3)
    :param logits (256, 256)
    """
    # convert to 1, 256, 256
    sam_model.set_image(frame)
    mode = "mask"
    logits = logits.unsqueeze(0)
    logits = TF.resize(logits, [256, 256]).cpu().numpy()
    prompts = {"mask_input": logits}  # 1 256 256
    masks, scores, logits = sam_model.predict(
        prompts, mode, multimask=True
    )  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    return masks, scores, logits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, default="horsejump-high")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/XMem-s012.pth")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    DATA_ROOT = "/shared/vye/datasets/DAVIS"
    # video frames (take videos from DAVIS-2017 as examples)
    img_paths = sorted(glob.glob(f"{DATA_ROOT}/JPEGImages/480p/{args.seq}/*.jpg"))
    # load frames
    frames = []
    for video_path in img_paths:
        frames.append(np.array(Image.open(video_path).convert("RGB")))
    frames = np.stack(frames, 0)  # T, H, W, C

    # load first frame annotation
    mask_paths = sorted(glob.glob(f"{DATA_ROOT}/Annotations/480p/{args.seq}/*.png"))
    assert len(mask_paths) == len(img_paths)
    first_frame_path = mask_paths[0]
    first_frame_annotation = np.array(
        Image.open(first_frame_path).convert("P")
    )  # H, W, each pixel is the class index
    num_classes = first_frame_annotation.max() + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    XMEM_checkpoint = "../checkpoints/XMem-s012.pth"
    tracker = BaseTracker(args.checkpoint, device)

    # for each frame, get tracking results by tracker.track(frame, first_frame_annotation)
    # frame: numpy array (H, W, C), first_frame_annotation: numpy array (H, W), leave it blank when tracking begins
    masks = []
    cmap = plt.get_cmap("gist_rainbow")
    os.makedirs(args.out_dir, exist_ok=True)
    writer = iio.get_writer(f"{args.out_dir}/{args.seq}_xmem_tracks.mp4", fps=args.fps)
    for ti, frame in tqdm(enumerate(frames)):
        if ti == 0:
            mask = tracker.track(frame, first_frame_annotation)
        else:
            mask = tracker.track(frame)
        masks.append(mask)
        mask_color = cmap(mask / num_classes)[..., :3]
        vis = frame / 255 * 0.4 + mask_color * 0.6
        writer.append_data((vis * 255).astype(np.uint8))
    writer.close()

    # clear memory in XMEM for the next video
    tracker.clear_memory()
