import colorsys
import datetime
import os
import subprocess

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np
import torch
from loguru import logger as guru
from mask_utils import init_sam_model, init_tracker, track_masks


class PromptGUI(object):
    def __init__(self, checkpoint_dir, sam_model_type, device):
        self.checkpoint_dir = checkpoint_dir
        self.sam_model_type = sam_model_type
        self.device = device
        self.sam_model = None
        self.tracker = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []

    def lazy_init_sam_model(self):
        if self.sam_model is None:
            self.sam_model = init_sam_model(
                self.checkpoint_dir, self.sam_model_type, self.device
            )

    def lazy_init_tracker(self):
        if self.tracker is None:
            self.tracker = init_tracker(self.checkpoint_dir, self.device)

    def clear_points(self) -> tuple[None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "Cleared points, select new points to update mask"
        return None, None, message

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        message = f"Creating new mask with index {self.cur_mask_idx}"
        return None, message

    def make_index_mask(self):
        assert len(self.cur_masks) > 0
        idcs = list(self.cur_masks.keys())
        idx_mask = self.cur_masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = self.cur_masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    def _clear_image(self):
        """
        clears image and all masks/logits for that image
        """
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self._clear_image()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image
        return image

    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        if self.image is None:
            return "Please select an image first", None
        self.lazy_init_sam_model()
        assert self.sam_model is not None
        self.sam_model.set_image(self.image)
        msg = (
            "SAM features extracted. "
            "Click points to update mask, and submit when ready to start tracking"
        )
        return msg, self.image

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        return "Selecting positive points. Submit the mask to start tracking"

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        return "Selecting negative points. Submit the mask to start tracking"

    def add_point(self, img, i, j):
        """
        get the index mask of the objects
        """
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # masks, scores, logits if we want to update the mask
        mask, logit = self.get_sam_mask(
            img, np.array(self.selected_points), np.array(self.selected_labels)
        )
        self.cur_masks[self.cur_mask_idx] = mask
        self.cur_logits[self.cur_mask_idx] = logit
        idx_mask = self.make_index_mask()
        return idx_mask

    def get_sam_mask(self, img, input_points, input_labels):
        """
        :param img (np array) (H, W, 3)
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) mask, (H, W) logits
        """
        self.lazy_init_sam_model()
        assert self.sam_model is not None
        if self.sam_model.is_image_set is False:
            self.sam_model.set_image(img)

        logits = self.cur_logits.get(self.cur_mask_idx, None)
        mask_input = None if logits is None else logits[None]
        masks, scores, logits = self.sam_model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=mask_input,
            multimask_output=True,
        )
        idx_sel = np.argmax(scores)
        return masks[idx_sel], logits[idx_sel]

    def run_tracker(self) -> tuple[str, str]:
        idx_mask = self.make_index_mask()
        self.lazy_init_tracker()
        assert self.tracker is not None
        self.tracker.clear_memory()

        images = [iio.imread(p) for p in self.img_paths]
        # binary masks
        self.index_masks_all = track_masks(
            self.tracker, images, idx_mask, self.frame_index
        )

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        message = f"Wrote current tracked video to {out_vidpath}."
        instruct = "Save the masks to an output directory if it looks good!"
        return out_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, output_dir: str) -> str:
        assert self.color_masks_all is not None
        os.makedirs(output_dir, exist_ok=True)
        for img_path, clr_mask in zip(self.img_paths, self.color_masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            iio.imwrite(out_path, clr_mask)
        message = f"Saved masks to {output_dir}!"
        guru.debug(message)
        return message


def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    """
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.5):
    max_idx = max([m.max() for m in index_masks])
    guru.debug(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u


def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        return sorted(os.listdir(vid_dir))
    return []


def make_demo(
    checkpoint_dir,
    sam_model_type,
    device,
    root_dir,
    vid_name: str = "videos",
    img_name: str = "images",
    mask_name: str = "masks",
):
    prompts = PromptGUI(checkpoint_dir, sam_model_type, device)

    start_instructions = (
        "Select a video file to extract frames from, "
        "or select an image directory with frames already extracted."
    )
    vid_root, img_root = (f"{root_dir}/{vid_name}", f"{root_dir}/{img_name}")
    with gr.Blocks() as demo:
        instruction = gr.Textbox(
            start_instructions, label="Instruction", interactive=False
        )
        with gr.Row():
            root_dir_field = gr.Text(root_dir, label="Dataset root directory")
            vid_name_field = gr.Text(vid_name, label="Video subdirectory name")
            img_name_field = gr.Text(img_name, label="Image subdirectory name")
            mask_name_field = gr.Text(mask_name, label="Mask subdirectory name")
            seq_name_field = gr.Text(None, label="Sequence name", interactive=False)

        with gr.Row():
            with gr.Column():
                vid_files = listdir(vid_root)
                vid_files_field = gr.Dropdown(label="Video files", choices=vid_files)
                input_video_field = gr.Video(label="Input Video")

                with gr.Row():
                    start_time = gr.Number(0, label="Start time (s)")
                    end_time = gr.Number(0, label="End time (s)")
                    sel_fps = gr.Number(30, label="FPS")
                    sel_height = gr.Number(540, label="Height")
                    extract_button = gr.Button("Extract frames")

            with gr.Column():
                img_dirs = listdir(img_root)
                img_dirs_field = gr.Dropdown(
                    label="Image directories", choices=img_dirs
                )
                img_dir_field = gr.Text(
                    None, label="Input directory", interactive=False
                )
                frame_index = gr.Slider(
                    label="Frame index",
                    minimum=0,
                    maximum=len(prompts.img_paths) - 1,
                    value=0,
                    step=1,
                )
                sam_button = gr.Button("Get SAM features")
                input_image = gr.Image(
                    prompts.set_input_image(0),
                    label="Input Frame",
                    every=1,
                )
                with gr.Row():
                    pos_button = gr.Button("Toggle positive")
                    neg_button = gr.Button("Toggle negative")
                clear_button = gr.Button("Clear points")

            with gr.Column():
                output_img = gr.Image(label="Current selection")
                add_button = gr.Button("Add new mask")
                submit_button = gr.Button("Submit mask for tracking")
                final_video = gr.Video(label="Masked video")
                mask_dir_field = gr.Text(
                    None, label="Path to save masks", interactive=False
                )
                save_button = gr.Button("Save masks")

        def update_vid_root(root_dir, vid_name):
            vid_root = f"{root_dir}/{vid_name}"
            vid_paths = listdir(vid_root)
            guru.debug(f"Updating video paths: {vid_paths=}")
            return vid_paths

        def update_img_root(root_dir, img_name):
            img_root = f"{root_dir}/{img_name}"
            img_dirs = listdir(img_root)
            guru.debug(f"Updating img dirs: {img_dirs=}")
            return img_root, img_dirs

        def update_mask_dir(root_dir, mask_name, seq_name):
            return f"{root_dir}/{mask_name}/{seq_name}"

        def update_root_paths(root_dir, vid_name, img_name, mask_name, seq_name):
            return (
                update_vid_root(root_dir, vid_name),
                update_img_root(root_dir, img_name),
                update_mask_dir(root_dir, mask_name, seq_name),
            )

        def select_video(root_dir, vid_name, seq_file):
            seq_name = os.path.splitext(seq_file)[0]
            guru.debug(f"Selected video: {seq_file=}")
            vid_path = f"{root_dir}/{vid_name}/{seq_file}"
            return seq_name, vid_path

        def extract_frames(
            root_dir, vid_name, img_name, vid_file, start, end, fps, height, ext="png"
        ):
            seq_name = os.path.splitext(vid_file)[0]
            vid_path = f"{root_dir}/{vid_name}/{vid_file}"
            out_dir = f"{root_dir}/{img_name}/{seq_name}"
            guru.debug(f"Extracting frames to {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            def make_time(seconds):
                return datetime.time(
                    seconds // 3600, (seconds % 3600) // 60, seconds % 60
                )

            start_time = make_time(start).strftime("%H:%M:%S")
            end_time = make_time(end).strftime("%H:%M:%S")
            cmd = (
                f"ffmpeg -ss {start_time} -to {end_time} -i {vid_path} "
                f"-vf 'scale=-1:{height},fps={fps}' {out_dir}/%05d.{ext}"
            )
            print(cmd)
            subprocess.call(cmd, shell=True)
            img_root = f"{root_dir}/{img_name}"
            img_dirs = listdir(img_root)
            return out_dir, img_dirs

        def select_image_dir(root_dir, img_name, seq_name):
            img_dir = f"{root_dir}/{img_name}/{seq_name}"
            guru.debug(f"Selected image dir: {img_dir}")
            return seq_name, img_dir

        def update_image_dir(root_dir, img_name, seq_name):
            img_dir = f"{root_dir}/{img_name}/{seq_name}"
            num_imgs = prompts.set_img_dir(img_dir)
            slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1)
            message = (
                f"Loaded {num_imgs} images from {img_dir}. Choose a frame to run SAM!"
            )
            return slider, message

        def get_select_coords(img, evt: gr.SelectData):
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            index_mask = prompts.add_point(img, i, j)
            guru.debug(f"{index_mask.shape=}")
            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(img, color_mask)
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out

        # update the root directory
        # and associated video, image, and mask root directories
        root_dir_field.submit(
            update_root_paths,
            [
                root_dir_field,
                vid_name_field,
                img_name_field,
                mask_name_field,
                seq_name_field,
            ],
            outputs=[vid_files_field, img_dirs_field, mask_dir_field],
        )
        vid_name_field.submit(
            update_vid_root,
            [root_dir_field, vid_name_field],
            outputs=[vid_files_field],
        )
        img_name_field.submit(
            update_img_root,
            [root_dir_field, img_name_field],
            outputs=[img_dirs_field],
        )
        mask_name_field.submit(
            update_mask_dir,
            [root_dir_field, mask_name_field, seq_name_field],
            outputs=[mask_dir_field],
        )

        # selecting a video file
        vid_files_field.select(
            select_video,
            [root_dir_field, vid_name_field, vid_files_field],
            outputs=[seq_name_field, input_video_field],
        )

        # when the img_dir_field changes
        img_dir_field.change(
            update_image_dir,
            [root_dir_field, img_name_field, seq_name_field],
            [frame_index, instruction],
        )
        seq_name_field.change(
            update_mask_dir,
            [root_dir_field, mask_name_field, seq_name_field],
            outputs=[mask_dir_field],
        )

        # selecting an image directory
        img_dirs_field.select(
            select_image_dir,
            [root_dir_field, img_name_field, img_dirs_field],
            [seq_name_field, img_dir_field],
        )

        # extracting frames from video
        extract_button.click(
            extract_frames,
            [
                root_dir_field,
                vid_name_field,
                img_name_field,
                vid_files_field,
                start_time,
                end_time,
                sel_fps,
                sel_height,
            ],
            outputs=[img_dir_field, img_dirs_field],
        )

        frame_index.change(prompts.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [input_image], [output_img])

        sam_button.click(prompts.get_sam_features, outputs=[instruction, input_image])
        clear_button.click(
            prompts.clear_points, outputs=[output_img, final_video, instruction]
        )
        pos_button.click(prompts.set_positive, outputs=[instruction])
        neg_button.click(prompts.set_negative, outputs=[instruction])

        add_button.click(prompts.add_new_mask, outputs=[output_img, instruction])
        submit_button.click(prompts.run_tracker, outputs=[final_video, instruction])
        save_button.click(
            prompts.save_masks_to_dir, [mask_dir_field], outputs=[instruction]
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--vid_name", type=str, default="videos")
    parser.add_argument("--img_name", type=str, default="images")
    parser.add_argument("--mask_name", type=str, default="masks")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = make_demo(
        args.checkpoint_dir,
        args.sam_model_type,
        device,
        args.root_dir,
        args.vid_name,
        args.img_name,
    )
    demo.launch(server_port=args.port)
