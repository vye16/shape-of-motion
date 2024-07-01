import argparse
import json
import os.path as osp
from glob import glob
from itertools import product

import cv2
import imageio.v3 as iio
import numpy as np
import roma
import torch
from tqdm import tqdm

from flow3d.metrics import mLPIPS, mPSNR, mSSIM
from flow3d.transforms import rt_to_mat4, solve_procrustes
from flow3d.data.colmap import get_colmap_camera_params

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the data directory that contains all the sequences.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    help="Path to the result directory that contains the results."
    "for batch evaluation, result_dir should contain subdirectories for each sequence. (result_dir/seq_name/results)"
    "for single sequence evaluation, result_dir should contain results directly (result_dir/results)",
)
parser.add_argument(
    "--seq_names",
    type=str,
    nargs="+",
    default=[
        "apple",
        "backpack",
        "block",
        "creeper",
        "handwavy",
        "haru-sit",
        "mochi-high-five",
        "paper-windmill",
        "pillow",
        "spin",
        "sriracha-tree",
        "teddy",
    ],
    help="Sequence names to evaluate.",
)
args = parser.parse_args()


def load_data_dict(data_dir, train_names, val_names):
    val_imgs = np.array(
        [iio.imread(osp.join(data_dir, "rgb/1x", f"{name}.png")) for name in val_names]
    )
    val_covisibles = np.array(
        [
            iio.imread(
                osp.join(
                    data_dir, "flow3d_preprocessed/covisible/1x/val/", f"{name}.png"
                )
            )
            for name in tqdm(val_names, desc="Loading val covisibles")
        ]
    )
    train_depths = np.array(
        [
            np.load(osp.join(data_dir, "depth/1x", f"{name}.npy"))[..., 0]
            for name in train_names
        ]
    )
    train_Ks, train_w2cs = get_colmap_camera_params(
        osp.join(data_dir, "flow3d_preprocessed/colmap/sparse/"),
        [name + ".png" for name in train_names],
    )
    train_Ks = train_Ks[:, :3, :3]
    scale = np.load(osp.join(data_dir, "flow3d_preprocessed/colmap/scale.npy")).item()
    train_c2ws = np.linalg.inv(train_w2cs)
    train_c2ws[:, :3, -1] *= scale
    train_w2cs = np.linalg.inv(train_c2ws)
    keypoint_paths = sorted(glob(osp.join(data_dir, "keypoint/2x/train/0_*.json")))
    keypoints_2d = []
    for keypoint_path in keypoint_paths:
        with open(keypoint_path) as f:
            keypoints_2d.append(json.load(f))
    keypoints_2d = np.array(keypoints_2d)
    keypoints_2d[..., :2] *= 2.0
    time_ids = np.array(
        [int(osp.basename(p).split("_")[1].split(".")[0]) for p in keypoint_paths]
    )
    time_pairs = np.array(list(product(time_ids, repeat=2)))
    index_pairs = np.array(list(product(range(len(time_ids)), repeat=2)))
    keypoints_3d = []
    for i, kps_2d in zip(time_ids, keypoints_2d):
        K = train_Ks[i]
        w2c = train_w2cs[i]
        depth = train_depths[i]
        is_kp_visible = kps_2d[:, 2] == 1
        is_depth_valid = (
            cv2.remap(
                (depth != 0).astype(np.float32),
                kps_2d[None, :, :2].astype(np.float32),
                None,  # type: ignore
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )[0]
            == 1
        )
        kp_depths = cv2.remap(
            depth,  # type: ignore
            kps_2d[None, :, :2].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        kps_3d = (
            np.einsum(
                "ij,pj->pi",
                np.linalg.inv(K),
                np.pad(kps_2d[:, :2], ((0, 0), (0, 1)), constant_values=1),
            )
            * kp_depths[:, None]
        )
        kps_3d = np.einsum(
            "ij,pj->pi",
            np.linalg.inv(w2c)[:3],
            np.pad(kps_3d, ((0, 0), (0, 1)), constant_values=1),
        )
        kps_3d = np.concatenate(
            [kps_3d, (is_kp_visible & is_depth_valid)[:, None]], axis=1
        )
        kps_3d[kps_3d[:, -1] != 1] = 0.0
        keypoints_3d.append(kps_3d)
    keypoints_3d = np.array(keypoints_3d)
    return {
        "val_imgs": val_imgs,
        "val_covisibles": val_covisibles,
        "train_depths": train_depths,
        "train_Ks": train_Ks,
        "train_w2cs": train_w2cs,
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "time_ids": time_ids,
        "time_pairs": time_pairs,
        "index_pairs": index_pairs,
    }


def load_result_dict(result_dir, val_names):
    try:
        pred_val_imgs = np.array(
            [iio.imread(osp.join(result_dir, "rgb", f"{name}.png")) for name in val_names]
            )
    except:
        pred_val_imgs = None
    try:
        keypoints_dict = np.load(
            osp.join(result_dir, "keypoints.npz"), allow_pickle=True
        )
        if len(keypoints_dict) == 1 and "arr_0" in keypoints_dict:
            keypoints_dict = keypoints_dict["arr_0"].item()
        pred_keypoint_Ks = keypoints_dict["Ks"]
        pred_keypoint_w2cs = keypoints_dict["w2cs"]
        pred_keypoints_3d = keypoints_dict["pred_keypoints_3d"]
        pred_train_depths = keypoints_dict["pred_train_depths"]
    except:
        print(
            "No keypoints.npz found, make sure that it's the method itself cannot produce keypoints."
        )
        keypoints_dict = {}
        pred_keypoint_Ks = None
        pred_keypoint_w2cs = None
        pred_keypoints_3d = None
        pred_train_depths = None

    if "visibilities" in list(keypoints_dict.keys()):
        pred_visibilities = keypoints_dict["visibilities"]
    else:
        pred_visibilities = None

    return {
        "pred_val_imgs": pred_val_imgs,
        "pred_train_depths": pred_train_depths,
        "pred_keypoint_Ks": pred_keypoint_Ks,
        "pred_keypoint_w2cs": pred_keypoint_w2cs,
        "pred_keypoints_3d": pred_keypoints_3d,
        "pred_visibilities": pred_visibilities,
    }


def evaluate_3d_tracking(data_dict, result_dict):
    train_Ks = data_dict["train_Ks"]
    train_w2cs = data_dict["train_w2cs"]
    keypoints_3d = data_dict["keypoints_3d"]
    time_ids = data_dict["time_ids"]
    time_pairs = data_dict["time_pairs"]
    index_pairs = data_dict["index_pairs"]
    pred_keypoint_Ks = result_dict["pred_keypoint_Ks"]
    pred_keypoint_w2cs = result_dict["pred_keypoint_w2cs"]
    pred_keypoints_3d = result_dict["pred_keypoints_3d"]
    if not np.allclose(train_Ks[time_ids], pred_keypoint_Ks):
        print("Inconsistent camera intrinsics.")
        print(train_Ks[time_ids][0], pred_keypoint_Ks[0])
    keypoint_w2cs = train_w2cs[time_ids]
    q, t, s = solve_procrustes(
        torch.from_numpy(np.linalg.inv(pred_keypoint_w2cs)[:, :3, -1]).to(
            torch.float32
        ),
        torch.from_numpy(np.linalg.inv(keypoint_w2cs)[:, :3, -1]).to(torch.float32),
    )[0]
    R = roma.unitquat_to_rotmat(q.roll(-1, dims=-1))
    pred_keypoints_3d = np.einsum(
        "ij,...j->...i",
        rt_to_mat4(R, t, s).numpy().astype(np.float64),
        np.pad(pred_keypoints_3d, ((0, 0), (0, 0), (0, 1)), constant_values=1),
    )
    pred_keypoints_3d = pred_keypoints_3d[..., :3] / pred_keypoints_3d[..., 3:]
    # Compute 3D tracking metrics.
    pair_keypoints_3d = keypoints_3d[index_pairs]
    is_covisible = (pair_keypoints_3d[:, :, :, -1] == 1).all(axis=1)
    target_keypoints_3d = pair_keypoints_3d[:, 1, :, :3]
    epes = []
    for i in range(len(time_pairs)):
        epes.append(
            np.linalg.norm(
                target_keypoints_3d[i][is_covisible[i]]
                - pred_keypoints_3d[i][is_covisible[i]],
                axis=-1,
            )
        )
    epe = np.mean(
        [frame_epes.mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_10cm = np.mean(
        [(frame_epes < 0.1).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_5cm = np.mean(
        [(frame_epes < 0.05).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    print(f"3D tracking EPE: {epe:.4f}")
    print(f"3D tracking PCK (10cm): {pck_3d_10cm:.4f}")
    print(f"3D tracking PCK (5cm): {pck_3d_5cm:.4f}")
    print("-----------------------------")
    return epe, pck_3d_10cm, pck_3d_5cm


def project(Ks, w2cs, pts):
    """
    Args:
        Ks: (N, 3, 3) camera intrinsics.
        w2cs: (N, 4, 4) camera extrinsics.
        pts: (N, N, M, 3) 3D points.
    """
    N = Ks.shape[0]
    pts = pts.swapaxes(0, 1).reshape(N, -1, 3)

    pts_homogeneous = np.concatenate([pts, np.ones_like(pts[..., -1:])], axis=-1)

    # Apply world-to-camera transformation
    pts_homogeneous = np.matmul(w2cs[:, :3], pts_homogeneous.swapaxes(1, 2)).swapaxes(
        1, 2
    )
    # Project to image plane using intrinsic parameters
    projected_pts = np.matmul(Ks, pts_homogeneous.swapaxes(1, 2)).swapaxes(1, 2)

    depths = projected_pts[..., 2:3]
    # Normalize homogeneous coordinates
    projected_pts = projected_pts[..., :2] / np.clip(depths, a_min=1e-6, a_max=None)
    projected_pts = projected_pts.reshape(N, N, -1, 2).swapaxes(0, 1)
    depths = depths.reshape(N, N, -1).swapaxes(0, 1)
    return projected_pts, depths


def evaluate_2d_tracking(data_dict, result_dict):
    train_w2cs = data_dict["train_w2cs"]
    keypoints_2d = data_dict["keypoints_2d"]
    visibilities = keypoints_2d[..., -1].astype(np.bool_)
    time_ids = data_dict["time_ids"]
    num_frames = len(time_ids)
    num_pts = keypoints_2d.shape[1]
    pred_train_depths = result_dict["pred_train_depths"]
    pred_keypoint_Ks = result_dict["pred_keypoint_Ks"]
    pred_keypoint_w2cs = result_dict["pred_keypoint_w2cs"]
    pred_keypoints_3d = result_dict["pred_keypoints_3d"].reshape(
        num_frames, -1, num_pts, 3
    )
    keypoint_w2cs = train_w2cs[time_ids]
    s = solve_procrustes(
        torch.from_numpy(np.linalg.inv(pred_keypoint_w2cs)[:, :3, -1]).to(
            torch.float32
        ),
        torch.from_numpy(np.linalg.inv(keypoint_w2cs)[:, :3, -1]).to(torch.float32),
    )[0][-1].item()

    target_points = keypoints_2d[None].repeat(num_frames, axis=0)[..., :2]
    target_visibilities = visibilities[None].repeat(num_frames, axis=0)

    pred_points, pred_depths = project(
        pred_keypoint_Ks, pred_keypoint_w2cs, pred_keypoints_3d
    )
    if result_dict["pred_visibilities"] is not None:
        pred_visibilities = result_dict["pred_visibilities"].reshape(
            num_frames, -1, num_pts
        )
    else:
        rendered_depths = []
        for i, points in zip(
            data_dict["index_pairs"][:, -1],
            pred_points.reshape(-1, pred_points.shape[2], 2),
        ):
            rendered_depths.append(
                cv2.remap(
                    pred_train_depths[i].astype(np.float32),
                    points[None].astype(np.float32),  # type: ignore
                    None,  # type: ignore
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )[0]
            )
        rendered_depths = np.array(rendered_depths).reshape(num_frames, -1, num_pts)
        pred_visibilities = (np.abs(rendered_depths - pred_depths) * s) < 0.05

    one_hot_eye = np.eye(target_points.shape[0])[..., None].repeat(num_pts, axis=-1)
    evaluation_points = one_hot_eye == 0
    for i in range(num_frames):
        evaluation_points[i, :, ~visibilities[i]] = False
    occ_acc = np.sum(
        np.equal(pred_visibilities, target_visibilities) & evaluation_points
    ) / np.sum(evaluation_points)
    all_frac_within = []
    all_jaccard = []

    for thresh in [4, 8, 16, 32, 64]:
        within_dist = np.sum(
            np.square(pred_points - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, target_visibilities)
        count_correct = np.sum(is_correct & evaluation_points)
        count_visible_points = np.sum(target_visibilities & evaluation_points)
        frac_correct = count_correct / count_visible_points
        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visibilities & evaluation_points)
        gt_positives = np.sum(target_visibilities & evaluation_points)
        false_positives = (~target_visibilities) & pred_visibilities
        false_positives = false_positives | ((~within_dist) & pred_visibilities)
        false_positives = np.sum(false_positives & evaluation_points)
        jaccard = true_positives / (gt_positives + false_positives)
        all_jaccard.append(jaccard)
    AJ = np.mean(all_jaccard)
    APCK = np.mean(all_frac_within)

    print(f"2D tracking AJ: {AJ:.4f}")
    print(f"2D tracking avg PCK: {APCK:.4f}")
    print(f"2D tracking occlusion accuracy: {occ_acc:.4f}")
    print("-----------------------------")
    return AJ, APCK, occ_acc


def evaluate_nv(data_dict, result_dict):
    device = "cuda"
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)

    val_imgs = torch.from_numpy(data_dict["val_imgs"])[..., :3].to(device)
    val_covisibles = torch.from_numpy(data_dict["val_covisibles"]).to(device)
    pred_val_imgs = torch.from_numpy(result_dict["pred_val_imgs"]).to(device)

    for i in range(len(val_imgs)):
        val_img = val_imgs[i] / 255.0
        pred_val_img = pred_val_imgs[i] / 255.0
        val_covisible = val_covisibles[i] / 255.0
        psnr_metric.update(val_img, pred_val_img, val_covisible)
        ssim_metric.update(val_img[None], pred_val_img[None], val_covisible[None])
        lpips_metric.update(val_img[None], pred_val_img[None], val_covisible[None])
    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    print(f"NV mPSNR: {mpsnr:.4f}")
    print(f"NV mSSIM: {mssim:.4f}")
    print(f"NV mLPIPS: {mlpips:.4f}")
    return mpsnr, mssim, mlpips


if __name__ == "__main__":
    seq_names = args.seq_names

    epe_all, pck_3d_10cm_all, pck_3d_5cm_all = [], [], []
    AJ_all, APCK_all, occ_acc_all = [], [], []
    mpsnr_all, mssim_all, mlpips_all = [], [], []

    for seq_name in seq_names:
        print("=========================================")
        print(f"Evaluating {seq_name}")
        print("=========================================")
        data_dir = osp.join(args.data_dir, seq_name)
        result_dir = osp.join(args.result_dir, seq_name, "results/")
        if not osp.exists(result_dir):
            result_dir = osp.join(args.result_dir, "results/")
        elif not osp.exists(result_dir):
            raise ValueError(f"Result directory {result_dir} not found.")

        with open(osp.join(data_dir, "splits/train.json")) as f:
            train_names = json.load(f)["frame_names"]
        with open(osp.join(data_dir, "splits/val.json")) as f:
            val_names = json.load(f)["frame_names"]

        data_dict = load_data_dict(data_dir, train_names, val_names)
        result_dict = load_result_dict(result_dir, val_names)
        if result_dict["pred_keypoints_3d"] is not None:
            epe, pck_3d_10cm, pck_3d_5cm = evaluate_3d_tracking(data_dict, result_dict)
            AJ, APCK, occ_acc = evaluate_2d_tracking(data_dict, result_dict)
            epe_all.append(epe)
            pck_3d_10cm_all.append(pck_3d_10cm)
            pck_3d_5cm_all.append(pck_3d_5cm)
            AJ_all.append(AJ)
            APCK_all.append(APCK)
            occ_acc_all.append(occ_acc)
        if len(data_dict["val_imgs"]) > 0:
            if result_dict["pred_val_imgs"] is None:
                print("No NV results found.")
                continue
            mpsnr, mssim, mlpips = evaluate_nv(data_dict, result_dict)
            mpsnr_all.append(mpsnr)
            mssim_all.append(mssim)
            mlpips_all.append(mlpips)

    print(f"mean 3D tracking EPE: {np.mean(epe_all):.4f}")
    print(f"mean 3D tracking PCK (10cm): {np.mean(pck_3d_10cm_all):.4f}")
    print(f"mean 3D tracking PCK (5cm): {np.mean(pck_3d_5cm_all):.4f}")
    print(f"mean 2D tracking AJ: {np.mean(AJ_all):.4f}")
    print(f"mean 2D tracking avg PCK: {np.mean(APCK_all):.4f}")
    print(f"mean 2D tracking occlusion accuracy: {np.mean(occ_acc_all):.4f}")
    print(f"mean NV mPSNR: {np.mean(mpsnr_all):.4f}")
    print(f"mean NV mSSIM: {np.mean(mssim_all):.4f}")
    print(f"mean NV mLPIPS: {np.mean(mlpips_all):.4f}")
