import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(
    image_dirs: list[str],
    gpus: list[int],
    image_name: str = "images",
    mask_name: str = "masks",
    metric_depth_name: str = "unidepth_disp",
    intrins_name: str = "unidepth_intrins",
    mono_depth_model: str = "depth-anything",
    slam_name: str = "droid_recon",
    track_model: str = "bootstapir",
):
    mono_depth_name = mono_depth_model.replace("-", "_")
    with ProcessPoolExecutor(max_workers=len(gpus)) as exc:
        for i, img_dir in enumerate(image_dirs):
            gpu = gpus[i % len(gpus)]
            img_dir = img_dir.rstrip("/")
            exc.submit(
                process_sequence,
                gpu,
                img_dir,
                img_dir.replace(image_name, mask_name),
                img_dir.replace(image_name, metric_depth_name),
                img_dir.replace(image_name, intrins_name),
                img_dir.replace(image_name, mono_depth_name),
                img_dir.replace(image_name, f"aligned_{mono_depth_name}"),
                img_dir.replace(image_name, slam_name),
                img_dir.replace(image_name, track_model),
                mono_depth_model,
                track_model,
            )


def process_sequence(
    gpu: int,
    img_dir: str,
    mask_dir: str,
    metric_depth_dir: str,
    intrins_name: str,
    mono_depth_dir: str,
    aligned_depth_dir: str,
    slam_path: str,
    track_dir: str,
    depth_model: str = "depth-anything",
    track_model: str = "bootstapir",
):
    dev_arg = f"CUDA_VISIBLE_DEVICES={gpu}"
    # XXX activate environments
    source_pre = "source /home/vye/anaconda3/bin/activate"
    # source_pre = "conda activate"
    metric_env = "unidepth"
    droid_env = "4dgs"
    track_env = "tapnet"
    our_env = "4dgs"

    metric_depth_cmd = (
        f"{dev_arg} python compute_metric_depth.py --img-dir {img_dir} "
        f"--depth-dir {metric_depth_dir} --intrins-file {intrins_name}.json"
    )
    cmd = f"{source_pre} {metric_env}; {metric_depth_cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    mono_depth_cmd = (
        f"{dev_arg} python compute_depth.py --img_dir {img_dir} "
        f"--out_raw_dir {mono_depth_dir} --out_aligned_dir {aligned_depth_dir} "
        f"--model {depth_model} --metric_dir {metric_depth_dir}"
    )
    cmd = f"source /home/hangg/.anaconda3/bin/activate {our_env}; {mono_depth_cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    slam_cmd = (
        f"{dev_arg} python recon_with_depth.py --image_dir {img_dir} "
        f"--calib {intrins_name}.json --depth_dir {aligned_depth_dir} --out_path {slam_path}"
    )
    cmd = f"source /home/hangg/.anaconda3/bin/activate {droid_env}; {slam_cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    slam_cmd = (
        f"{dev_arg} python recon_with_depth.py --image_dir {img_dir} "
        f"--calib {intrins_name}.json --depth_dir {aligned_depth_dir} --out_path {slam_path}"
    )
    cmd = f"{source_pre} {droid_env}; {slam_cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    track_cmd = (
        f"{dev_arg} python compute_tracks_pairwise.py --image_dir {img_dir} "
        f"--mask_dir {mask_dir} --out_dir {track_dir} --model_type {track_model}"
    )
    cmd = f"{source_pre} {track_env}; {track_cmd}"
    print(cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    tyro.cli(main)
