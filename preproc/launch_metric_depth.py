import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    depth_name: str = "unidepth_disp",
    intrins_name: str = "unidepth_intrins",
):
    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting {img_name} in {img_dirs[0]}")

    with ProcessPoolExecutor(max_workers=len(gpus)) as exe:
        for i, img_dir in enumerate(img_dirs):
            if not os.path.isdir(img_dir):
                print(f"Skipping {img_dir} as it is not a directory")
                continue
            dev_id = gpus[i % len(gpus)]
            depth_dir = img_dir.replace(img_name, depth_name)
            intrins_file = f"{img_dir.replace(img_name, intrins_name)}.json"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={dev_id} python compute_metric_depth.py "
                f"--img-dir {img_dir} --depth-dir {depth_dir} --intrins-file {intrins_file}"
            )
            print(cmd)
            exe.submit(subprocess.call, cmd, shell=True)


if __name__ == "__main__":
    tyro.cli(main)
