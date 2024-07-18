import os
import subprocess

import tyro


def extract_frames(
    video_path: str,
    output_root: str,
    height: int,
    ext: str,
    skip_time: int = 1,
    start_time: str = "00:00:00",
    end_time: str | None = None,
):
    seq_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, seq_name)
    os.makedirs(output_dir, exist_ok=True)
    to_str = f"-to {end_time}" if end_time else ""
    command = f"ffmpeg -i {video_path} -vf \"select='not(mod(n,{skip_time}))',scale=-1:{height}\" -vsync vfr -ss {start_time} {to_str} {output_dir}/%05d.{ext}"
    subprocess.call(command, shell=True)


def main(
    video_paths: list[str],
    output_root: str,
    height: int = 540,
    ext: str = "jpg",
    skip_time: int = 1,
    start_time: str = "00:00:00",
    end_time: str | None = None,
):
    for video_path in video_paths:
        extract_frames(
            video_path, output_root, height, ext, skip_time, start_time, end_time
        )


if __name__ == "__main__":
    tyro.cli(main)
