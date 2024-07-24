import json
import os

import imageio.v3 as iio
import numpy as np
import torch
import tyro
from tqdm import tqdm
from unidepth.models import UniDepthV1


def run_model_inference(img_dir: str, depth_dir: str, intrins_file: str):
    img_files = sorted(os.listdir(img_dir))
    if not intrins_file.endswith(".json"):
        intrins_file = f"{intrins_file}.json"

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(os.path.dirname(intrins_file), exist_ok=True)
    if len(os.listdir(depth_dir)) == len(img_files) and os.path.isfile(intrins_file):
        print(
            f"found {len(img_files)} files in {depth_dir}, found {intrins_file}, skipping"
        )
        return

    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Torch version:", torch.__version__)
    print(f"Running on {img_dir} with {len(img_files)} images")
    model = model.to(device)
    intrins_dict = {}
    for img_file in (bar := tqdm(img_files)):
        img_name = os.path.splitext(img_file)[0]
        out_path = f"{depth_dir}/{img_name}.npy"
        img = iio.imread(f"{img_dir}/{img_file}")
        pred_dict = run_model(model, img)
        depth = pred_dict["depth"]
        disp = 1.0 / np.clip(depth, a_min=1e-6, a_max=1e6)
        bar.set_description(f"Input {img_file} {depth.min()} {depth.max()}")
        np.save(out_path.replace("png", "npy"), disp.squeeze())

        K = pred_dict["intrinsics"]
        intrins_dict[img_name] = (
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )

    with open(intrins_file, "w") as f:
        json.dump(intrins_dict, f, indent=1)


def run_model(model, rgb: np.ndarray, intrinsics: np.ndarray | None = None):
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics_torch = None
    if intrinsics is not None:
        intrinsics_torch = torch.from_numpy(intrinsics)

    predictions = model.infer(rgb_torch, intrinsics_torch)
    out_dict = {k: v.squeeze().cpu().numpy() for k, v in predictions.items()}
    return out_dict


if __name__ == "__main__":
    tyro.cli(run_model_inference)
