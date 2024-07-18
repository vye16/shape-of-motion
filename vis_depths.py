import sys
from time import sleep
from typing import Annotated, Union

import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from tqdm import tqdm
from viser import transforms as vtf

from flow3d.data import DavisDataConfig, get_train_val_datasets, iPhoneDataConfig
from flow3d.vis.utils import get_server


def main(
    data: Union[
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")],
        Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")],
    ],
    port: int = 8890,
):
    guru.remove()
    guru.add(sys.stdout, level="INFO")

    dset, _, _, _ = get_train_val_datasets(data, load_val=False)

    server = get_server(port)
    bg_points, _, bg_colors = dset.get_bkgd_points(10000)
    print(f"{bg_points.shape=}")
    server.scene.add_point_cloud(
        "bg_points", bg_points.numpy(), bg_colors.numpy(), point_size=0.01
    )

    T = dset.num_frames
    depth = dset.get_depth(0)
    H, W = depth.shape[:2]
    r = 2
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, W, r, dtype=torch.float32),
            torch.arange(0, H, r, dtype=torch.float32),
            indexing="xy",
        ),
        dim=-1,
    )
    Ks = dset.get_Ks()
    fx = Ks[0, 0, 0]
    fov = float(2 * torch.atan(0.5 * W / fx))
    w2cs = dset.get_w2cs()
    print(f"{grid.shape=} {depth[::r,::r].shape=}")

    all_points, all_colors = [], []
    for i in tqdm(range(T)):
        img = dset.get_image(i)[::r, ::r]
        depth = dset.get_depth(i)[::r, ::r]
        mask = dset.get_mask(i)[::r, ::r]
        bool_mask = (mask != 0) & (depth > 0)
        K = Ks[i]
        w2c = w2cs[i]

        points = (
            torch.einsum(
                "ij,pj->pi",
                torch.linalg.inv(K),
                F.pad(grid[bool_mask], (0, 1), value=1.0),
            )
            * depth[bool_mask][:, None]
        )
        points = torch.einsum(
            "ij,pj->pi",
            torch.linalg.inv(w2c)[:3],
            F.pad(points, (0, 1), value=1.0),
        ).reshape(-1, 3)
        clrs = img[bool_mask].reshape(-1, 3)
        all_points.append(points)
        all_colors.append(clrs)

    while True:
        for w2c, points, clrs in zip(w2cs, all_points, all_colors):
            cam_tf = vtf.SE3.from_matrix(w2c.numpy()).inverse()
            wxyz, pos = cam_tf.wxyz_xyz[:4], cam_tf.wxyz_xyz[4:]
            server.scene.add_camera_frustum(
                "camera", fov=fov, aspect=W / H, wxyz=wxyz, position=pos
            )
            server.scene.add_point_cloud(
                "points", points.numpy(), clrs.numpy(), point_size=0.01
            )
            sleep(0.3)


if __name__ == "__main__":
    tyro.cli(main)
