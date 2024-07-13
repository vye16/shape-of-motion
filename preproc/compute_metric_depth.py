import json
import os

import imageio.v3 as iio
import numpy as np
import torch
import tyro
from tqdm import tqdm
from unidepth.models import UniDepthV1


def write_depth(path, depth, rescale: bool, bits: int = 2):
    """
    Write depth map to png file.
    :param path (str): filepath without extension
    :param depth (array): depth
    """
    assert bits in [1, 2], "Unsupported bit depth."

    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    max_val = (2 ** (8 * bits)) - 1
    dtype = "uint8" if bits == 1 else "uint16"

    if rescale:
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
    else:  # save metric depth
        out = depth * 256 if bits == 2 else depth
    out = out.astype(dtype).squeeze()

    iio.imwrite(path, out)


def run_model_inference(img_dir: str, depth_dir: str, intrins_file: str):
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_files = sorted(os.listdir(img_dir))
    if not intrins_file.endswith(".json"):
        intrins_file = f"{intrins_file}.json"
    if len(os.listdir(depth_dir)) == len(img_files) and os.path.isfile(intrins_file):
        print(
            f"found {len(img_files)} files in {depth_dir}, found {intrins_file}, skipping"
        )
        return

    print("Torch version:", torch.__version__)
    print(f"Running on {img_dir} with {len(img_files)} images")
    model = model.to(device)
    intrins_dict = {}
    for img_file in (bar := tqdm(img_files)):
        img_name = os.path.splitext(img_file)[0]
        out_path = f"{depth_dir}/{img_name}.png"
        bar.set_description(f"Input {img_file} output {out_path}")
        img = iio.imread(f"{img_dir}/{img_file}")
        pred_dict = run_model(model, img)
        depth = pred_dict["depth"]
        write_depth(out_path, depth, False)

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
