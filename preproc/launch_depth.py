import os
import tyro
import subprocess
from concurrent.futures import ProcessPoolExecutor


def main(
    img_dirs: list[str],
    depth_root: str,
    aligned_root: str,
    devices: list[int],
    metric_root: str | None = None,
    sparse_root: str | None = None,
    depth_model: str = "depth-anything-v2",
):
    with ProcessPoolExecutor(max_workers=len(devices)) as exe:
        for i, img_dir in enumerate(img_dirs):
            if not os.path.isdir(img_dir):
                print(f"Skipping {img_dir} as it is not a directory")
                continue
            dev_id = devices[i % len(devices)]
            seq_name = os.path.basename(img_dir.rstrip("/"))
            depth_dir = os.path.join(depth_root, seq_name)
            aligned_dir = os.path.join(aligned_root, seq_name)
            ref_arg = ""
            if metric_root is not None:
                metric_dir = os.path.join(metric_root, seq_name)
                ref_arg = f"--metric_dir {metric_dir}"
            elif sparse_root is not None:
                sparse_dir = os.path.join(sparse_root, seq_name)
                ref_arg = f"--sparse_dir {sparse_dir}"
            os.makedirs(aligned_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            cmd = (
                f"CUDA_VISIBLE_DEVICES={dev_id} python compute_depth.py "
                f"--img_dir {img_dir} --out_raw_dir {depth_dir} "
                f"--out_aligned_dir {aligned_dir} {ref_arg} "
                f"--model {depth_model}"
            )
            exe.submit(subprocess.call, cmd, shell=True)

if __name__ == "__main__":
    tyro.cli(main)
