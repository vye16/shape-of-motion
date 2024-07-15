import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import tyro


def main(
    devices: list[int],
    seqs: list[str] | None,
    work_root: str,
    davis_root: str = "/shared/vye/datasets/DAVIS",
    image_name: str = "JPEGImages",
    res: str = "480p",
    depth_type: str = "aligned_depth_anything",
):
    img_dir = f"{davis_root}/{image_name}/{res}"
    if seqs is None:
        seqs = sorted(os.listdir(img_dir))
    with ProcessPoolExecutor() as exc:
        for i, seq_name in enumerate(seqs):
            device = devices[i % len(devices)]
            cmd = (
                f"CUDA_VISIBLE_DEVICES={device} python run_training.py "
                f"--work-dir {work_root}/{seq_name} data:davis "
                f"--data.seq_name {seq_name} --data.root_dir {davis_root} "
                f"--data.res {res} --data.depth_type {depth_type}"
            )
            print(cmd)
            exc.submit(subprocess.call, cmd, shell=True)


if __name__ == "__main__":
    tyro.cli(main)
