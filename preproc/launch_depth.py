import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    metric_name: str | None = None,
    sparse_name: str | None = None,
    depth_model: str = "depth-anything-v2",
):
    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting {img_name} in {img_dirs[0]}")

    with ProcessPoolExecutor(max_workers=len(gpus)) as exe:
        for i, img_dir in enumerate(img_dirs):
            if not os.path.isdir(img_dir):
                print(f"Skipping {img_dir} as it is not a directory")
                continue
            dev_id = gpus[i % len(gpus)]
            depth_name = depth_model.replace("-", "_")
            depth_dir = img_dir.replace(img_name, depth_name)
            aligned_dir = img_dir.replace(img_name, f"aligned_{depth_name}")

            ref_arg = ""
            if metric_name is not None:
                metric_dir = img_dir.replace(img_name, metric_name)
                ref_arg = f"--metric_dir {metric_dir}"
            if sparse_name is not None:
                sparse_dir = img_dir.replace(img_name, sparse_name)
                ref_arg = f"--sparse_dir {sparse_dir}"
            cmd = (
                f"CUDA_VISIBLE_DEVICES={dev_id} python compute_depth.py "
                f"--img_dir {img_dir} --out_raw_dir {depth_dir} "
                f"--out_aligned_dir {aligned_dir} {ref_arg} "
                f"--model {depth_model}"
            )
            print(cmd)
            exe.submit(subprocess.call, cmd, shell=True)


if __name__ == "__main__":
    tyro.cli(main)
