import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    mask_name: str = "masks",
    model_type: str = "bootstapir",
    use_torch: bool = True,
):
    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting {img_name} in {img_dirs[0]}")

    script_name = "compute_tracks_torch.py" if use_torch else "compute_tracks_jax.py"
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        for i, img_dir in enumerate(img_dirs):
            gpu = gpus[i % len(gpus)]
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python {script_name} "
                f"--model_type {model_type} "
                f"--image_dir {img_dir} "
                f"--mask_dir {img_dir.replace(img_name, mask_name)} "
                f"--out_dir {img_dir.replace(img_name, model_type)} "
            )
            print(cmd)
            executor.submit(subprocess.run, cmd, shell=True)


if __name__ == "__main__":
    tyro.cli(main)
