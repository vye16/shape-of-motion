import argparse
import fnmatch
import os
import os.path as osp
from glob import glob
from typing import Protocol

import imageio.v2 as iio
import numpy as np
import torch
from pycolmap import SceneManager
from tqdm import tqdm
from flow3d.data.davis_dataset import load_cameras

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Predictor(Protocol):
    def infer_tensor(
        self, image: torch.Tensor, K: torch.Tensor | None
    ) -> torch.Tensor: ...


def get_predictor() -> Predictor:
    predictor = torch.hub.load("hangg7/dsine", "DSINE", trust_repo=True).to(DEVICE)
    print(f"dsine predictor loaded.")
    return predictor


@torch.inference_mode()
def get_dsine_normal(
    predictor: Predictor,
    img_file: str,
    K: np.ndarray | torch.Tensor | None = None,
):
    image = iio.imread(img_file)
    image = torch.as_tensor(
        image[..., :3].transpose(2, 0, 1)[None] / 255.0,
        dtype=torch.float32,
        device=DEVICE,
    )
    if K is not None:
        K = torch.as_tensor(K[None], dtype=torch.float32, device=DEVICE)
    normal = predictor.infer_tensor(image, K)
    normal = normal[0].cpu().numpy().transpose(1, 2, 0)
    return normal


def save_normal_from_dir_colmap(
    img_dir: str,
    out_dir: str,
    sparse_dir: str,
    matching_pattern: str = "*",
):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]

    manager = SceneManager(sparse_dir)
    manager.load()
    camdata = manager.cameras
    imdata = manager.images
    name_to_imdata = {imdata[im_id].name: imdata[im_id] for im_id in imdata}
    camdata = [camdata[name_to_imdata[osp.basename(f)].camera_id] for f in img_files]

    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.png"))) == len(img_files):
        print(f"dsine normal maps already computed for {img_dir}.")
        return

    predictor = get_predictor()
    os.makedirs(out_dir, exist_ok=True)
    for img_file, cam in zip(tqdm(img_files, f"computing dsine normal maps"), camdata):
        K = cam.get_camera_matrix()
        normal = get_dsine_normal(predictor, img_file, K)
        out_file = osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".png")
        iio.imwrite(out_file, ((normal + 1.0) / 2.0 * 255.0).astype(np.uint8))


def save_normal_from_dir_droid(
    img_dir: str,
    out_dir: str,
    droid_path: str,
    matching_pattern: str = "*",
):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]
    H, W = iio.imread(img_files[0]).shape[:2]

    predictor = get_predictor()
    os.makedirs(out_dir, exist_ok=True)

    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.png"))) == len(img_files):
        print(f"dsine normal maps already computed for {img_dir}.")
        return

    cameras = load_cameras(droid_path, H, W)[1]
    cameras = cameras.numpy()
    for img_file, cam in zip(tqdm(img_files, f"computing dsine normal maps"), cameras):
        K = cam
        normal = get_dsine_normal(predictor, img_file, K)
        out_file = osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".png")
        iio.imwrite(out_file, ((normal + 1.0) / 2.0 * 255.0).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sparse_dir", type=str, default=None)
    parser.add_argument("--droid_path", type=str, default=None)
    parser.add_argument("--matching_pattern", type=str, default="*")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.sparse_dir is not None:
        save_normal_from_dir_colmap(
            args.img_dir, args.out_dir, args.sparse_dir, args.matching_pattern
        )
    if args.droid_path is not None:
        save_normal_from_dir_droid(
            args.img_dir, args.out_dir, args.droid_path, args.matching_pattern
        )


if __name__ == "__main__":
    """
    
    """
    main()
