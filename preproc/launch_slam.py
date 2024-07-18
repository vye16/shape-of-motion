import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "images",
    depth_method: str = "aligned_depth_anything",
    intrins_method: str = "unidepth_intrins",
    out_name: str = "droid_recon",
):
    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting {img_name} in {img_dirs[0]}")

    print(f"Processing {len(img_dirs)} sequences")
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        for i, img_dir in enumerate(img_dirs):
            gpu = gpus[i % len(gpus)]
            depth_dir = img_dir.replace(img_name, depth_method)
            calib_path = f"{img_dir.replace(img_name, intrins_method)}.json"
            out_path = img_dir.replace(img_name, out_name)
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python recon_with_depth.py --img_dir {img_dir} "
                f"--calib {calib_path} --depth_dir {depth_dir} --out_path {out_path}"
            )
            print(cmd)
            executor.submit(subprocess.call, cmd, shell=True)


if __name__ == "__main__":
    tyro.cli(main)
