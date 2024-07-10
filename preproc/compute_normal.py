import argparse
import fnmatch
import os
import os.path as osp
from glob import glob
from typing import Literal

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_predictor():
    predictor = torch.hub.load(
        "/admin/home-hangg/projects/dsine", "DSINE", source="local", trust_repo=True
    ).to(DEVICE)
    print(f"dsine predictor loaded.")
    return predictor


def get_dsine_normal(
    predictor: Pipeline,
    img_file: str,
    ret_type: Literal["uint16", "float"] = "float",
):

    image = Image.open(img_file)
    disp = pipe(image)["predicted_depth"]
    disp = torch.nn.functional.interpolate(
        disp.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
    )
    disp = disp.squeeze().cpu().numpy()
    if ret_type == "uint16":
        return to_uint16(disp)
    elif ret_type == "float":
        return disp
    else:
        raise ValueError(f"Unknown return type {ret_type}")


def save_disp_from_dir(
    model_name: str,
    img_dir: str,
    out_dir: str,
    matching_pattern: str = "*",
):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]
    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.png"))) == len(img_files):
        print(f"Raw {model_name} depth maps already computed for {img_dir}")
        return

    pipe = get_pipeline(model_name)
    os.makedirs(out_dir, exist_ok=True)
    for img_file in tqdm(img_files, f"computing {model_name} depth maps"):
        disp = get_depth_anything_disp(pipe, img_file, ret_type="uint16")
        out_file = osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".png")
        iio.imwrite(out_file, disp)


def align_monodepth_with_colmap(
    sparse_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    manager = SceneManager(sparse_dir)
    manager.load()

    cameras = manager.cameras
    images = manager.images
    points3D = manager.points3D
    point3D_id_to_point3D_idx = manager.point3D_id_to_point3D_idx

    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    os.makedirs(output_monodepth_dir, exist_ok=True)
    images = [
        image
        for _, image in images.items()
        if fnmatch.fnmatch(image.name, matching_pattern)
    ]
    for image in tqdm(images, "Aligning monodepth with colmap point cloud"):

        point3D_ids = image.point3D_ids
        point3D_ids = point3D_ids[point3D_ids != manager.INVALID_POINT3D]
        pts3d_valid = points3D[[point3D_id_to_point3D_idx[id] for id in point3D_ids]]  # type: ignore
        K = cameras[image.camera_id].get_camera_matrix()
        rot = image.R()
        trans = image.tvec.reshape(3, 1)
        extrinsics = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)

        pts3d_valid_homo = np.concatenate(
            [pts3d_valid, np.ones_like(pts3d_valid[..., :1])], axis=-1
        )
        pts3d_valid_cam_homo = extrinsics.dot(pts3d_valid_homo.T).T
        pts2d_valid_cam = K.dot(pts3d_valid_cam_homo[..., :3].T).T
        pts2d_valid_cam = pts2d_valid_cam[..., :2] / pts2d_valid_cam[..., 2:3]
        colmap_depth = pts3d_valid_cam_homo[..., 2]

        monodepth_path = osp.join(
            input_monodepth_dir, osp.splitext(image.name)[0] + ".png"
        )
        mono_disp_map = iio.imread(monodepth_path) / 65535.0

        colmap_disp = 1.0 / np.clip(colmap_depth, a_min=1e-6, a_max=1e6)
        mono_disp = cv2.remap(
            mono_disp_map,  # type: ignore
            pts2d_valid_cam[None, ...].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        ms_colmap_disp = colmap_disp - np.median(colmap_disp) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(colmap_disp - scale * mono_disp)

        mono_disp_aligned = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(mono_disp_aligned, 0.01))
        # set depth values that are too small to invalid (0)
        mono_disp_aligned[mono_disp_aligned < min_thre] = 0.0
        np.save(
            osp.join(output_monodepth_dir, image.name.split(".")[0] + ".npy"),
            mono_disp_aligned,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything",
        help="depth model to use, one of [depth-anything, depth-anything-v2]",
    )
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_raw_dir", type=str, required=True)
    parser.add_argument("--out_aligned_dir", type=str, required=True)
    parser.add_argument("--sparse_dir", type=str, required=True)
    parser.add_argument("--matching_pattern", type=str, default="*")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert args.model in [
        "depth-anything",
        "depth-anything-v2",
    ], f"Unknown model {args.model}"
    save_disp_from_dir(
        args.model, args.img_dir, args.out_raw_dir, args.matching_pattern
    )
    align_monodepth_with_colmap(
        args.sparse_dir, args.out_raw_dir, args.out_aligned_dir, args.matching_pattern
    )


if __name__ == "__main__":
    """ example usage for iphone dataset:
    python compute_depth.py \
        --img_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/rgb/1x \
        --out_raw_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/depth_anything_v2 \
        --out_aligned_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/aligned_depth_anything_v2 \
        --sparse_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/colmap/sparse \
        --matching_pattern "0_*"
    """
    main()

#  device = "cuda"

#  dsine_model = torch.hub.load(
#      "/admin/home-hangg/projects/dsine", "DSINE", source="local", trust_repo=True
#  ).to(device)
