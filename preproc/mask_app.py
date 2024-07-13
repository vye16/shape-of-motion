import colorsys
import os

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np
import torch
import tyro
from loguru import logger as guru
from mask_utils import init_sam_model, init_tracker, track_masks


class PromptGUI(object):
    def __init__(
        self, checkpoint_dir, sam_model_type, device, img_dir: str | None = None
    ):
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
        self.cur_mask = None
        self.cur_logits = None
        self.masks_all = None

        if img_dir is not None:
            self.set_img_dir(img_dir)
        else:
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

    def clear_points(self) ->tuple[None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "Cleared points, select new points to update mask"
        return None, None, message

    def _clear_image(self):
        self.image = None
        self.cur_mask = None
        self.cur_logits = None
        self.masks_all = None

    def set_img_dir(
        self, img_dir: str
    ) -> tuple[np.ndarray | None, str, gr.Slider, str]:
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        slider = gr.Slider(minimum=0, maximum=len(self.img_paths) - 1, value=0, step=1)
        message = f"Loaded {len(self.img_paths)} images from {img_dir}. Choose a frame to run SAM!"
        return self.set_input_image(0), img_dir, slider, message

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

    def get_sam_features(self) -> str:
        if self.image is None:
            return "Please select an image first"
        self.lazy_init_sam_model()
        assert self.sam_model is not None
        self.sam_model.set_image(self.image)
        return (
            "SAM features extracted. "
            "Click points to update mask, and submit when ready to start tracking"
        )

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        return "Selecting positive points. Submit the mask to start tracking"

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        return "Selecting negative points. Submit the mask to start tracking"

    def add_point(self, img, i, j):
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # masks, scores, logits if we want to update the mask
        mask, logit = self.get_sam_mask(
            img, np.array(self.selected_points), np.array(self.selected_labels)
        )
        self.cur_mask, self.cur_logits = mask, logit
        return self.cur_mask

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
        mask_input = None if self.cur_logits is None else self.cur_logits[None]
        masks, scores, logits = self.sam_model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=mask_input,
            multimask_output=True,
        )
        idx_sel = np.argmax(scores)
        return masks[idx_sel], logits[idx_sel]

    def run_tracker(self) -> tuple[str, str]:
        assert self.cur_mask is not None
        self.lazy_init_tracker()
        assert self.tracker is not None
        images = [iio.imread(p) for p in self.img_paths]
        self.masks_all = track_masks(
            self.tracker, images, self.cur_mask, self.frame_index
        )
        out_frames = colorize_masks(images, self.masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        message = f"Wrote current tracked video to {out_vidpath}."
        instruct = "Save the masks to an output directory if it looks good!"
        return out_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, output_dir: str) -> str:
        assert self.masks_all is not None
        os.makedirs(output_dir, exist_ok=True)
        for img_path, mask in zip(self.img_paths, self.masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            iio.imwrite(out_path, mask.astype("uint8") * 255)
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


def colorize_masks(images, masks_all):
    color = np.array([30, 144, 255]) / 255
    out_frames = []
    for img, mask in zip(images, masks_all):
        color_masks = color * mask.astype("int")[..., None]
        out_f = 0.5 * img / 255 + 0.5 * color_masks
        out_u = (255 * out_f).astype("uint8")
        out_frames.append(out_u)
    return out_frames


def get_palette(num_colors: int, lightness: float = 0.5, saturation: float = 0.8):
    hues = np.linspace(0, 1, int(num_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    return [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]


def make_demo(checkpoint_dir, sam_model_type, device, img_dir: str | None = None):
    prompts = PromptGUI(checkpoint_dir, sam_model_type, device, img_dir=img_dir)
    mask_col = np.array([30, 144, 255]) / 255

    start_instructions = (
        "Select the frame you want to use"
        if img_dir is not None
        else "Enter path to the image directory"
    )
    with gr.Blocks() as demo:
        instruction = gr.Textbox(
            start_instructions, label="Instruction", interactive=False
        )
        with gr.Row():
            with gr.Column():
                input_dir = gr.Text(img_dir, label="Input directory")
                with gr.Row():
                    clear_button = gr.Button("Clear points")
                    pos_button = gr.Button("Positive points")
                    neg_button = gr.Button("Negative points")
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

            with gr.Column():
                output_img = gr.Image(label="Current selection")
                submit_button = gr.Button("Submit mask for tracking")
                final_video = gr.Video(label="Masked video")
                with gr.Row():
                    save_dir_input = gr.Text(
                        img_dir, label="Path to save masks", scale=3
                    )
                    save_button = gr.Button("Save masks")
                    # save_output = gr.Text(scale=1)

        def get_select_coords(img, evt: gr.SelectData):
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            mask = prompts.add_point(img, i, j)
            col_mask = mask[..., None] * mask_col
            out_f = 0.5 * img / 255 + 0.5 * col_mask
            out_u = (255 * out_f).astype("uint8")
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out

        sam_button.click(prompts.get_sam_features, outputs=[instruction])
        clear_button.click(
            prompts.clear_points, outputs=[output_img, final_video, instruction]
        )
        pos_button.click(prompts.set_positive, outputs=[instruction])
        neg_button.click(prompts.set_negative, outputs=[instruction])
        submit_button.click(prompts.run_tracker, outputs=[final_video, instruction])
        save_button.click(
            prompts.save_masks_to_dir, [save_dir_input], outputs=[instruction]
        )

        input_dir.submit(
            prompts.set_img_dir,
            [input_dir],
            [input_image, save_dir_input, frame_index, instruction],
        )
        frame_index.change(prompts.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [input_image], [output_img])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--img_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = make_demo(args.checkpoint_dir, args.sam_model_type, device, args.img_dir)
    demo.launch(server_port=args.port)
