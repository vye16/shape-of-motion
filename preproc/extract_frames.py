import tyro
import os
import subprocess


def extract_frames(video_path: str, output_root: str, height: int, ext: str):
    seq_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, seq_name)
    os.makedirs(output_dir, exist_ok=True)
    command = f"ffmpeg -i {video_path} -vf 'scale=-1:{height}' {output_dir}/%05d.{ext}"
    subprocess.call(command, shell=True)


def main(video_paths: list[str], output_root: str, height: int = 540, ext: str = "jpg"):
    for video_path in video_paths:
        extract_frames(video_path, output_root, height, ext)


if __name__ == "__main__":
    tyro.cli(main)
