import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(img_dirs: list[str], depth_root: str, intrins_root: str, devices: list[int]):
    os.makedirs(intrins_root, exist_ok=True)

    with ProcessPoolExecutor(max_workers=len(devices)) as exe:
        for i, img_dir in enumerate(img_dirs):
            if not os.path.isdir(img_dir):
                print(f"Skipping {img_dir} as it is not a directory")
                continue
            dev_id = devices[i % len(devices)]
            seq_name = os.path.basename(img_dir.rstrip("/"))
            depth_dir = os.path.join(depth_root, seq_name)
            os.makedirs(depth_dir, exist_ok=True)
            intrins_file = f"{intrins_root}/{seq_name}.json"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={dev_id} python compute_metric_depth.py "
                f"--img-dir {img_dir} --depth-dir {depth_dir} --intrins-file {intrins_file}"
            )
            exe.submit(subprocess.call, cmd, shell=True)


if __name__ == "__main__":
    tyro.cli(main)
