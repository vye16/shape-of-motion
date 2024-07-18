import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor


def launch_batch(
    device: int,
    root_dir: str,
    seq_name: str,
    img_name: str = "images",
    depth_method: str = "aligned_depth_anything",
    intrins_method: str = "unidepth_intrins",
    out_name: str = "droid_recon",
    res: str = "480p",
):
    image_dir = f"{root_dir}/{img_name}/{res}/{seq_name}"
    depth_dir = f"{root_dir}/{depth_method}/{res}/{seq_name}"
    calib_path = f"{root_dir}/{intrins_method}/{res}/{seq_name}.json"
    out_path = f"{root_dir}/{out_name}/{seq_name}"
    cmd = (
        f"CUDA_VISIBLE_DEVICES={device} python recon_with_depth.py --image_dir {image_dir} "
        f"--calib {calib_path} --depth_dir {depth_dir} --out_path {out_path}"
    )
    print(cmd)
    subprocess.call(cmd, shell=True)


def main(args):
    if args.dataset == "davis":
        args.img_name = "JPEGImages"
        args.res = "480p"
    else:
        args.res = ""

    image_root = f"{args.root_dir}/{args.img_name}/{args.res}"
    seq_names = os.listdir(image_root)
    if args.seq_names is not None:
        seq_names = args.seq_names

    print(f"Processing {len(seq_names)} sequences")
    with ProcessPoolExecutor(max_workers=len(args.devices)) as executor:
        for i, seq_name in enumerate(seq_names):
            device = args.devices[i % len(args.devices)]
            executor.submit(
                launch_batch,
                device,
                args.root_dir,
                seq_name,
                args.img_name,
                args.depth_method,
                args.intrins_method,
                args.out_name,
                args.res,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=[0])
    parser.add_argument(
        "--dataset", type=str, choices=["davis", "custom"], help="dataset type"
    )
    parser.add_argument("--root_dir", type=str, help="path to dataset directory")
    parser.add_argument("--img_name", type=str, default="images")
    parser.add_argument("--depth_method", type=str, default="aligned_depth_anything")
    parser.add_argument("--intrins_method", type=str, default="unidepth_intrins")
    parser.add_argument("--out_name", type=str, default="droid_recon")
    parser.add_argument("--seq_names", nargs="+", default=None)
    parser.add_argument("--res", type=str, default="480p")
    args = parser.parse_args()
    main(args)
