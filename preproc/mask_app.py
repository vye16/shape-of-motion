import colorsys
import os
from typing import Tuple

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
        self.sam_model = init_sam_model(checkpoint_dir, sam_model_type, device)
        self.tracker = init_tracker(checkpoint_dir, device)

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

    def clear_points(self):
        self.selected_points.clear()
        self.selected_labels.clear()

    def clear_image(self):
        self.image = None
        self.cur_mask = None
        self.cur_logits = None
        self.masks_all = None

    def set_img_dir(self, img_dir: str) -> Tuple[np.ndarray, str]:
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        return self.set_input_image(0), img_dir

    def set_input_image(self, i: int) -> np.ndarray:
        self.clear_points()
        self.clear_image()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.sam_model.set_image(image)
        self.image = image
        return image

    def set_positive(self):
        self.cur_label_val = 1.0

    def set_negative(self):
        self.cur_label_val = 0.0

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

    def run_tracker(self):
        assert self.cur_mask is not None
        images = [iio.imread(p) for p in self.img_paths]
        self.masks_all = track_masks(
            self.tracker, images, self.cur_mask, self.frame_index
        )
        out_frames = colorize_masks(images, self.masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        return out_vidpath

    def save_masks_to_dir(self, output_dir):
        assert self.masks_all is not None
        os.makedirs(output_dir, exist_ok=True)
        for img_path, mask in zip(self.img_paths, self.masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            iio.imwrite(out_path, mask.astype("uint8") * 255)
        message = f"Saved masks to {output_dir}"
        guru.info(message)
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

    with gr.Blocks() as demo:
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
                input_img = gr.Image(prompts.set_input_image(0), label="Input", every=1)

            with gr.Column():
                output_img = gr.Image(label="Output")
                submit_button = gr.Button("Submit")
                final_video = gr.Video(label="Masked video")
                with gr.Row():
                    save_dir_input = gr.Text(
                        img_dir, label="Path to save masks", scale=3
                    )
                    save_output = gr.Text(scale=1)

        def get_select_coords(img, evt: gr.SelectData):
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            mask = prompts.add_point(img, i, j)
            col_mask = mask[..., None] * mask_col
            out_f = 0.5 * img / 255 + 0.5 * col_mask
            out_u = (255 * out_f).astype("uint8")
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out

        clear_button.click(prompts.clear_points, outputs=[output_img, final_video])
        pos_button.click(prompts.set_positive)
        neg_button.click(prompts.set_negative)
        submit_button.click(prompts.run_tracker, outputs=final_video)
        save_dir_input.submit(prompts.save_masks_to_dir, [save_dir_input], save_output)

        frame_index.change(prompts.set_input_image, [frame_index], input_img)
        input_dir.submit(prompts.set_img_dir, [input_dir], [input_img, save_dir_input])
        input_img.select(get_select_coords, [input_img], output_img)

    return demo


def main(
    port: int = 8890,
    checkpoint_dir: str = "checkpoints",
    sam_model_type: str = "vit_h",
    img_dir: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = make_demo(checkpoint_dir, sam_model_type, device, img_dir)
    demo.launch(server_port=port)


if __name__ == "__main__":
    tyro.cli(main)
