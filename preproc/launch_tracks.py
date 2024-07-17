import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor


def main(args):
    if args.dataset == "davis":
        img_root = f"{args.data_root}/JPEGImages/{args.res}"
        mask_root = f"{args.data_root}/Annotations/{args.res}"
        out_root = f"{args.data_root}/{args.model_type}/{args.res}"
    else:
        img_root = f"{args.data_root}/{args.image_name}"
        mask_root = f"{args.data_root}/{args.mask_name}"
        out_root = f"{args.data_root}/{args.model_type}"

    launch_batch(
        args.gpus, args.seq_names, img_root, mask_root, out_root, args.model_type
    )


def launch_batch(gpus, seq_names, img_root, mask_root, out_root, model_type):
    if len(seq_names) == 0:
        seq_names = sorted(os.listdir(img_root))

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        for i, seq_name in enumerate(seq_names):
            gpu = gpus[i % len(gpus)]
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python compute_tracks_pairwise.py "
                f"--model_type {model_type} "
                f"--image_dir {img_root}/{seq_name} "
                f"--mask_dir {mask_root}/{seq_name} "
                f"--out_dir {out_root}/{seq_name}"
            )
            print(cmd)
            executor.submit(subprocess.run, cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="data root")
    parser.add_argument(
        "--seq_names", nargs="*", type=str, default=[], help="seq names"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["bootstapir", "tapir"],
        default="bootstapir",
        help="which tapir checkpoint to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="davis",
        choices=["davis", "custom"],
        help="method",
    )
    parser.add_argument("--img_name", type=str, default="images", help="image name")
    parser.add_argument("--mask_name", type=str, default="masks", help="image name")
    parser.add_argument("--res", type=str, default="480p", help="resolution")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    main(args)
