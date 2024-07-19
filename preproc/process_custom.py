import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    mask_name: str = "masks",
    metric_depth_name: str = "unidepth_disp",
    intrins_name: str = "unidepth_intrins",
    mono_depth_model: str = "depth-anything",
    slam_name: str = "droid_recon",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
):
    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting {img_name} in {img_dirs[0]}")

    mono_depth_name = mono_depth_model.replace("-", "_")
    with ProcessPoolExecutor(max_workers=len(gpus)) as exc:
        for i, img_dir in enumerate(img_dirs):
            gpu = gpus[i % len(gpus)]
            img_dir = img_dir.rstrip("/")
            exc.submit(
                process_sequence,
                gpu,
                img_dir,
                img_dir.replace(img_name, mask_name),
                img_dir.replace(img_name, metric_depth_name),
                img_dir.replace(img_name, intrins_name),
                img_dir.replace(img_name, mono_depth_name),
                img_dir.replace(img_name, f"aligned_{mono_depth_name}"),
                img_dir.replace(img_name, slam_name),
                img_dir.replace(img_name, track_model),
                mono_depth_model,
                track_model,
                tapir_torch,
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
    tapir_torch: bool = True,
):
    dev_arg = f"CUDA_VISIBLE_DEVICES={gpu}"

    metric_depth_cmd = (
        f"{dev_arg} python compute_metric_depth.py --img-dir {img_dir} "
        f"--depth-dir {metric_depth_dir} --intrins-file {intrins_name}.json"
    )
    subprocess.call(metric_depth_cmd, shell=True, executable="/bin/bash")

    mono_depth_cmd = (
        f"{dev_arg} python compute_depth.py --img_dir {img_dir} "
        f"--out_raw_dir {mono_depth_dir} --out_aligned_dir {aligned_depth_dir} "
        f"--model {depth_model} --metric_dir {metric_depth_dir}"
    )
    print(mono_depth_cmd)
    subprocess.call(mono_depth_cmd, shell=True, executable="/bin/bash")

    slam_cmd = (
        f"{dev_arg} python recon_with_depth.py --img_dir {img_dir} "
        f"--calib {intrins_name}.json --depth_dir {aligned_depth_dir} --out_path {slam_path}"
    )
    print(slam_cmd)
    subprocess.call(slam_cmd, shell=True, executable="/bin/bash")

    track_script = "compute_tracks_torch.py" if tapir_torch else "compute_tracks_jax.py"
    track_cmd = (
        f"{dev_arg} python {track_script} --image_dir {img_dir} "
        f"--mask_dir {mask_dir} --out_dir {track_dir} --model_type {track_model}"
    )
    subprocess.call(track_cmd, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    tyro.cli(main)
