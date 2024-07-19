import argparse
import functools
import glob
import os

import haiku as hk
import imageio
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import tree
from tapnet.models import tapir_model
from tapnet.utils import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, required=True, help="image dir")
parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
parser.add_argument("--out_dir", type=str, required=True, help="out dir")
parser.add_argument("--grid_size", type=int, default=4, help="grid size")
parser.add_argument("--resize_height", type=int, default=256, help="resize height")
parser.add_argument("--resize_width", type=int, default=256, help="resize width")
parser.add_argument("--num_points", type=int, default=200, help="num points")
parser.add_argument(
    "--model_type", type=str, choices=["tapir", "bootstapir"], help="model type"
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="checkpoints",
    help="checkpoint dir",
)
args = parser.parse_args()

## Load model
ckpt_file = (
    "tapir_checkpoint_panning.npy"
    if args.model_type == "tapir"
    else "bootstapir_checkpoint_v2.npy"
)
ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)

ckpt_state = np.load(ckpt_path, allow_pickle=True).item()
params, state = ckpt_state["params"], ckpt_state["state"]


def init_model(model_type):
    if model_type == "bootstapir":
        model = tapir_model.TAPIR(
            bilinear_interp_with_depthwise_conv=False,
            pyramid_level=1,
            extra_convs=True,
            softmax_temperature=10.0,
        )
    else:
        model = tapir_model.TAPIR(
            bilinear_interp_with_depthwise_conv=False, pyramid_level=0
        )
    return model


def build_model(frames, query_points):
    """Compute point tracks and occlusions given frames and query points."""
    model = init_model(args.model_type)
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


def build_model_init(frames):
    model = init_model(args.model_type)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    return feature_grids


def build_model_predict(frames, points, feature_grids):
    """Compute point tracks and occlusions given frames and query points."""
    model = init_model(args.model_type)
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


resize_height = args.resize_height
resize_width = args.resize_width
num_points = args.num_points
grid_size = args.grid_size

folder_path = args.image_dir
mask_dir = args.mask_dir
frame_names = [
    os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
]
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

done = True
for t in range(len(frame_names)):
    for j in range(len(frame_names)):
        name_t = os.path.splitext(frame_names[t])[0]
        name_j = os.path.splitext(frame_names[j])[0]
        out_path = f"{out_dir}/{name_t}_{name_j}.npy"
        if not os.path.exists(out_path):
            done = False
            break
print(f"{done=}")
if done:
    print("Already done")
    exit()

video = read_video(folder_path)
num_frames, height, width = video.shape[0:3]
masks = read_video(mask_dir)
masks = (masks.reshape((num_frames, height, width, -1)) > 0).any(axis=-1)
print(f"{video.shape=} {masks.shape=} {masks.max()=} {masks.sum()=}")

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

for t in tqdm(range(num_frames), desc="frames"):
    name_t = os.path.splitext(frame_names[t])[0]
    file_matches = glob.glob(f"{out_dir}/{name_t}_*.npy")
    if len(file_matches) == num_frames:
        print(f"Already computed tracks with query {t=} {name_t=}")
        continue

    all_points = np.stack([t * np.ones_like(y), y_resize, x_resize], axis=-1)
    mask = masks[t]
    in_mask = mask[y, x] > 0.5
    all_points_t = all_points[in_mask]
    print(f"{all_points.shape=} {all_points_t.shape=} {t=}")
    outputs = []
    if len(all_points_t) > 0:
        num_chunks = max(1, len(all_points_t) // 128)
        for points in tqdm(
            np.array_split(all_points_t, axis=0, indices_or_sections=num_chunks),
            leave=False,
            desc="points",
        ):
            points = points.astype(np.float32)[None]  # Add batch dimension
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
    else:
        outputs = np.zeros((0, num_frames, 4), dtype=np.float32)

    for j in range(num_frames):
        if j == t:
            original_query_points = np.stack([x[in_mask], y[in_mask]], axis=-1)
            outputs[:, j, :2] = original_query_points
        name_j = os.path.splitext(frame_names[j])[0]
        np.save(f"{out_dir}/{name_t}_{name_j}.npy", outputs[:, j])
