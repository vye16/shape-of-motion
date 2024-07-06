import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import subprocess


def main(args):
    if args.dataset == "davis":
        launch_davis(args)
    elif args.dataset == "kubric":
        launch_kubric(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def launch_davis(args):
    seq_names = args.seq_names
    img_root = f"{args.data_root}/JPEGImages/{args.res}"
    mask_root = f"{args.data_root}/Annotations/{args.res}"
    out_root = f"{args.data_root}/2d_tracks/{args.res}"
    if len(seq_names) == 0:
        seq_names = sorted(os.listdir(img_root))

    with ProcessPoolExecutor(max_workers=len(args.gpus)) as executor:
        for i, seq_name in enumerate(seq_names):
            gpu = args.gpus[i % len(args.gpus)]
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python compute_tracks_pairwise.py "
                f"--image_dir {img_root}/{seq_name} "
                f"--mask_dir {mask_root}/{seq_name} "
                f"--out_dir {out_root}/{seq_name}"
            )
            executor.submit(subprocess.run, cmd, shell=True)


def launch_kubric(args):
    seq_names = args.seq_names
    img_root = f"{args.data_root}/images"
    mask_root = f"{args.data_root}/masks"
    out_root = f"{args.data_root}/2d_tracks"
    if len(seq_names) == 0:
        seq_names = sorted(os.listdir(img_root))

    with ProcessPoolExecutor(max_workers=len(args.gpus)) as executor:
        for i, seq_name in enumerate(seq_names):
            gpu = args.gpus[i % len(args.gpus)]
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python compute_tracks_faster.py "
                f"--image_dir {img_root}/{seq_name} "
                f"--mask_dir {mask_root}/{seq_name} "
                f"--out_dir {out_root}/{seq_name}"
            )
            executor.submit(subprocess.run, cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="data root")
    parser.add_argument(
        "--seq_names", nargs="*", type=str, default=[], help="seq names"
    )
    parser.add_argument("--dataset", type=str, default="davis", help="method")
    parser.add_argument("--res", type=str, default="480p", help="resolution")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    main(args)
