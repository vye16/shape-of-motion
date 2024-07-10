import os

basedir = os.path.dirname(os.path.abspath(__file__))
tap_dir = os.path.join(basedir, "tapnet")

import sys

sys.path.extend([tap_dir, basedir])
import argparse
import functools
import glob
import os
import subprocess

import haiku as hk
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tree
from tapir_model import QueryFeatures
from tapnet import tapir_model
from tapnet.utils import transforms, viz_utils
from tqdm import tqdm

# from google.colab import output
# output.enable_custom_widget_manager()


def gen_grid_np(h, w, normalize=False, homogeneous=False):
    if normalize:
        lin_y = np.linspace(-1.0, 1.0, num=h)
        lin_x = np.linspace(-1.0, 1.0, num=w)
    else:
        lin_y = np.arange(0, h)
        lin_x = np.arange(0, w)
    grid_x, grid_y = np.meshgrid(lin_x, lin_y)
    grid = np.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = np.concatenate([grid, np.ones_like(grid[..., :1])], axis=-1)
    return grid  # [h, w, 2 or 3]


checkpoint_path = "checkpoints/tapir_checkpoint_panning.npy"
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state["params"], ckpt_state["state"]


def build_model(frames, query_points):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(
        bilinear_interp_with_depthwise_conv=False, pyramid_level=0
    )
    outputs = model(
        video=frames,
        is_training=False,
        query_points=query_points,
        query_chunk_size=64,
    )
    return outputs


model = hk.transform_with_state(build_model)
model_apply = jax.jit(model.apply)


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32
      expected_dist: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    visibles = (1 - jax.nn.sigmoid(occlusions)) * (
        1 - jax.nn.sigmoid(expected_dist)
    ) > 0.5
    return visibles


def inference(frames, query_points):
    """Inference on one video.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8
      query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
      tracks: [num_points, 3], [-1, 1], [t, y, x]
      visibles: [num_points, num_frames], bool
    """
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    tracks, occlusions, expected_dist = (
        outputs["tracks"],
        outputs["occlusion"],
        outputs["expected_dist"],
    )

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles


def build_model_init(frames):
    model = tapir_model.TAPIR(
        bilinear_interp_with_depthwise_conv=False, pyramid_level=0
    )
    feature_grids = model.get_feature_grids(frames, is_training=False)
    return feature_grids


def build_model_predict(frames, points, feature_grids):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(
        bilinear_interp_with_depthwise_conv=False, pyramid_level=0
    )
    features = model.get_query_features(
        frames,
        is_training=False,
        query_points=points,
        feature_grids=feature_grids,
    )
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=features,
        query_points_in_video=points,
        query_chunk_size=128,
    )
    # return {k: v[-1] for k, v in trajectories.items()}
    p = model.num_pips_iter
    out = dict(
        occlusion=jnp.mean(jnp.stack(trajectories["occlusion"][p::p]), axis=0),
        tracks=jnp.mean(jnp.stack(trajectories["tracks"][p::p]), axis=0),
        expected_dist=jnp.mean(jnp.stack(trajectories["expected_dist"][p::p]), axis=0),
        unrefined_occlusion=trajectories["occlusion"][:-1],
        unrefined_tracks=trajectories["tracks"][:-1],
        unrefined_expected_dist=trajectories["expected_dist"][:-1],
    )
    return out


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
    return points


def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.stack([imageio.imread(frame_path) for frame_path in frame_paths])
    print(f"{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    video = media._VideoArray(video)
    return video


# seq_name = 'lab-coat'
# data_dir = '/home/qw246/data/davis_tapir/{}'.format(seq_name)
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True, help="image dir")
parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
parser.add_argument("--out_dir", type=str, required=True, help="out dir")
parser.add_argument("--grid_size", type=int, default=8, help="grid size")
parser.add_argument("--resize_height", type=int, default=256, help="resize height")
parser.add_argument("--resize_width", type=int, default=256, help="resize width")
parser.add_argument("--num_points", type=int, default=200, help="num points")
args = parser.parse_args()

resize_height = args.resize_height
resize_width = args.resize_width
num_points = args.num_points
grid_size = args.grid_size

folder_path = args.image_dir
mask_dir = args.mask_dir
frame_names = [
    os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
]
video = read_video(folder_path)
num_frames, height, width = video.shape[0:3]
masks = read_video(mask_dir)
masks = (masks.reshape((num_frames, height, width, -1)) > 0).any(axis=-1)
print(f"{video.shape=} {masks.shape=} {masks.max()=} {masks.sum()=}")

# data_dir = args.data_dir
# out_dir = os.path.join(data_dir, "2d_tracks")
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

frames = media.resize_video(video, (resize_height, resize_width))
print(f"{frames.shape=}")
frames = preprocess_frames(frames)[None]
print(f"preprocessed {frames.shape=}")

y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]
y_resize, x_resize = y / (height - 1) * (resize_height - 1), x / (width - 1) * (
    resize_width - 1
)

model_init = hk.transform_with_state(build_model_init)
model_init_apply = jax.jit(model_init.apply)

model_predict = hk.transform_with_state(build_model_predict)
model_predict_apply = jax.jit(model_predict.apply)

rng = jax.random.PRNGKey(42)
model_init_apply = functools.partial(
    model_init_apply, params=params, state=state, rng=rng
)
model_predict_apply = functools.partial(
    model_predict_apply, params=params, state=state, rng=rng
)

query_points = np.zeros([20, 3], dtype=np.float32)[None]
feature_grids, _ = model_init_apply(frames=frames)
print(f"{frames.shape=} {query_points.shape=}")

prediction, _ = model_predict_apply(
    frames=frames,
    points=query_points,
    feature_grids=feature_grids,
)

# todo convert it to flow!
# grid = gen_grid_np(height, width)

for t in tqdm(range(num_frames), desc="frames"):
    all_points = np.stack([t * np.ones_like(y), y_resize, x_resize], axis=-1)
    mask = masks[t]
    in_mask = mask[y, x] > 0.5
    all_points_t = all_points[in_mask]
    print(f"{all_points.shape=} {all_points_t.shape=} {t=}")
    outputs = []
    for points in tqdm(
        np.array_split(
            all_points_t, axis=0, indices_or_sections=len(all_points_t) // 128
        ),
        leave=False,
        desc="points",
    ):
        points = points.astype(np.float32)[None]  # Add batch dimension
        # print(f"{points.shape=}")
        prediction, _ = model_predict_apply(
            frames=frames,
            points=points,
            feature_grids=feature_grids,
        )
        prediction = tree.map_structure(lambda x: np.array(x[0]), prediction)
        track, occlusion, expected_dist = (
            prediction["tracks"],
            prediction["occlusion"],
            prediction["expected_dist"],
        )
        track = transforms.convert_grid_coordinates(
            track, (resize_width - 1, resize_height - 1), (width - 1, height - 1)
        )
        outputs.append(
            np.concatenate(
                [track, occlusion[..., None], expected_dist[..., None]], axis=-1
            )
        )

    outputs = np.concatenate(outputs, axis=0)
    for j in range(num_frames):
        if j == t:
            original_query_points = np.stack([x[in_mask], y[in_mask]], axis=-1)
            outputs[:, j, :2] = original_query_points
        name_t = os.path.splitext(frame_names[t])[0]
        name_j = os.path.splitext(frame_names[j])[0]
        np.save(f"{out_dir}/{name_t}_{name_j}.npy", outputs[:, j])
