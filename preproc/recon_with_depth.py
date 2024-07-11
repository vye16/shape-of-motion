import sys

import os
basedir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.dirname(basedir)
src_dir = os.path.join(basedir, "DROID-SLAM")
droid_dir = os.path.join(src_dir, "droid_slam")
sys.path.extend([src_dir, droid_dir])

import json
from tqdm import tqdm
import numpy as np
import torch
from lietorch import SE3
import cv2
import glob
import time
import argparse
import imageio.v2 as iio

from torch.multiprocessing import Process
from droid import Droid
import droid_backends

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", image / 255.0)
    cv2.waitKey(1)


def make_intrinsics(fx, fy, cx, cy):
    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    return K


def preproc_image(image, calib):
    if len(calib) > 4:
        fx, fy, cx, cy = calib[:4]
        K = make_intrinsics(fx, fy, cx, cy)
        image = cv2.undistort(image, K, calib[4:])

    h0, w0 = image.shape[:2]
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]
    return image, (h0, w0), (h1, w1)


def image_stream(image_dir, calib_path, stride, depth_dir: str | None = None):
    """image generator"""

    with open(calib_path, "r") as f:
        calib_dict = json.load(f)

    img_path_list = sorted(os.listdir(image_dir))[::stride]

    # give all images the same calibration
    calibs = torch.tensor([calib_dict[os.path.splitext(im)[0]] for im in img_path_list])
    calib = calibs.mean(dim=0)
    image = cv2.imread(os.path.join(image_dir, img_path_list[0]))
    image, (H0, W0), (H1, W1) = preproc_image(image, calib)

    fx, fy, cx, cy = calib.tolist()[:4]
    intrins = torch.as_tensor([fx, fy, cx, cy])
    intrins[0::2] *= W1 / W0
    intrins[1::2] *= H1 / H0

    for t, imfile in enumerate(img_path_list):
        imname = os.path.splitext(imfile)[0]
        image = cv2.imread(os.path.join(image_dir, imfile))
        image, (h0, w0), (h1, w1) = preproc_image(image, calib)
        assert h0 == H0 and w0 == W0 and h1 == H1 and w1 == W1
        image = torch.as_tensor(image).permute(2, 0, 1)

        if depth_dir is not None:
            depth_path = f"{depth_dir}/{imname}.png"
            depth = iio.imread(depth_path) / 256
            depth, (dh0, dw0), (dh1, dw1) = preproc_image(depth, calib)
            assert dh0 == h0 and dw0 == w0 and dh1 == h1 and dw1 == w1
            depth = torch.as_tensor(depth).float()

            yield t, image[None], intrins, depth
        else:
            yield t, image[None], intrins


def save_reconstruction(droid, traj_est, out_path, filter_thresh: float = 0.5, vis: bool = False):

    from pathlib import Path

    video = droid.video
    T = video.counter.value
    tstamps = video.tstamp[:T].cpu().numpy()
    (dirty_index,) = torch.where(video.dirty.clone())
    poses = torch.index_select(video.poses, 0, dirty_index)
    disps = torch.index_select(video.disps, 0, dirty_index)
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
    count = droid_backends.depth_filter(
        poses, disps, video.intrinsics[0], dirty_index, thresh
    )
    masks = (count >= 2) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True))

    points = (
        droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0])
        .cpu()
        .numpy()
    )
    map_c2w = SE3(poses).inv().data.cpu().numpy()
    masks = masks.cpu().numpy()
    images = video.images[:T].cpu()[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
    images = images.numpy()
    img_shape = images.shape[1:3]
    disps = disps.cpu().numpy()
    intrinsics = video.intrinsics[0].cpu().numpy()
    print(f"{points.shape=} {images.shape=} {masks.shape=} {map_c2w.shape=}")
    print(f"{img_shape=} {intrinsics=}")

    if vis:
        import viser

        server = viser.ViserServer(port=8890)
        handles = []
        for t in range(T):
            m = masks[t]
            print(f"{m.shape=} {m.sum()=}")
            pts = points[t][m]
            clrs = images[t][m]
            print(f"{pts.shape=} {clrs.shape=}")
            pc_h = server.add_point_cloud(f"frame_{t}", pts, clrs, point_size=0.05)
            trans = map_c2w[t, :3]
            quat = map_c2w[t, 3:]
            cam_h = server.add_camera_frustum(
                f"cam_{t}", fov=90, aspect=1, position=trans, wxyz=quat
            )
            handles.append((cam_h, pc_h))

        try:
            while True:
                for t in range(T):
                    for i, (cam_h, pc_h) in enumerate(handles):
                        if i != t:
                            pc_h.visible = False
                            cam_h.visible = False
                        else:
                            pc_h.visible = True
                            cam_h.visible = True
                    time.sleep(0.3)
        except KeyboardInterrupt:
            pass
    map_c2w_mat = SE3(torch.as_tensor(map_c2w)).matrix().numpy()
    traj_c2w_mat = SE3(torch.as_tensor(traj_est)).matrix().numpy()

    os.makedirs(os.path.dirname(out_path.rstrip("/")), exist_ok=True)
    save_dict = {
        "tstamps": tstamps,
        "images": images,
        "points": points,
        "masks": masks,
        "map_c2w": map_c2w_mat,
        "traj_c2w": traj_c2w_mat,
        "intrinsics": intrinsics,
        "img_shape": img_shape,
    }
    for k, v in save_dict.items():
        print(f"{k} {v.shape if isinstance(v, np.ndarray) else v}")
    np.save(out_path, np.array(save_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="path to image directory")
    parser.add_argument(
        "--depth_dir", type=str, default=None, help="path to depth directory"
    )
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default=f"{rootdir}/checkpoints/droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true", default=True)

    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="weight for translation / rotation components of flow",
    )
    parser.add_argument(
        "--filter_thresh",
        type=float,
        default=0.8,
        help="how much motion before considering new keyframe",
    )
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument(
        "--keyframe_thresh",
        type=float,
        default=4.0,
        help="threshold to create a new keyframe",
    )
    parser.add_argument(
        "--frontend_thresh",
        type=float,
        default=16.0,
        help="add edges between frames whithin this distance",
    )
    parser.add_argument(
        "--frontend_window", type=int, default=25, help="frontend optimization window"
    )
    parser.add_argument(
        "--frontend_radius",
        type=int,
        default=2,
        help="force edges between frames within radius",
    )
    parser.add_argument(
        "--frontend_nms", type=int, default=1, help="non-maximal supression of edges"
    )

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--out_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method("spawn")

    droid = None

    # need high resolution depths
    if args.out_path is not None:
        args.upsample = True

    tstamps = []
    for t, image, intrinsics, depth in tqdm(
        image_stream(args.image_dir, args.calib, args.stride, depth_dir=args.depth_dir)
    ):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        print(f"{t=} {image.shape=} {depth.shape if depth is not None else None}")
        droid.track(t, image, depth=depth, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.image_dir, args.calib, args.stride))

    if args.out_path is not None:
        save_reconstruction(droid, traj_est, args.out_path)
