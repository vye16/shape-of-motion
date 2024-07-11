from time import sleep
import torch
import torch.nn.functional as F
import tyro

from typing import Annotated, Union
from flow3d.data import (
    get_train_val_datasets,
    iPhoneDataConfig,
    DavisDataConfig,
)
from flow3d.vis.utils import get_server


def main(
    data: Union[
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")],
        Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")],
    ]
):
    dset, _, _, _ = get_train_val_datasets(data, load_val=False)

    server = get_server()
    T = dset.num_frames
    depth = dset.get_depth(0)
    H, W = depth.shape[:2]
    step = 1
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, W, step, dtype=torch.float32),
            torch.arange(0, H, step, dtype=torch.float32),
            indexing="xy",
        ),
        dim=-1,
    )
    Ks = dset.get_Ks()
    w2cs = dset.get_w2cs()
    print(f"{grid.shape=} {depth[::step,::step].shape=}")

    while True:
        for i in range(T):
            img = dset.get_image(i)[::step, ::step]
            depth = dset.get_depth(i)[::step, ::step]
            print(f"{depth.min()=} {depth.max()=}")
            mask = dset.get_mask(i)[::step, ::step]
            bool_mask = (mask != 0) & (depth > 0)
            # bool_mask = depth > 0
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
            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(w2c)[:3],
                    F.pad(points, (0, 1), value=1.0),
                )
                .reshape(-1, 3)
                .numpy()
            )
            clrs = img[bool_mask].reshape(-1, 3).numpy()
            server.scene.add_point_cloud("points", points, clrs, point_size=0.01)
            sleep(0.3)


if __name__ == "__main__":
    tyro.cli(main)
