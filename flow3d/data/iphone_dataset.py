import json
import os
import os.path as osp
from dataclasses import dataclass
from glob import glob
from itertools import product
from typing import Literal

import imageio.v3 as iio
import numpy as np
import roma
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from torch.utils.data import Dataset
from tqdm import tqdm

from flow3d.data.base_dataset import BaseDataset
from flow3d.data.colmap import get_colmap_camera_params
from flow3d.data.utils import (
    SceneNormDict,
    masked_median_blur,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4


@dataclass
class iPhoneDataConfig:
    data_dir: str
    start: int = 0
    end: int = -1
    split: Literal["train", "val"] = "train"
    depth_type: Literal[
        "midas",
        "depth_anything",
        "lidar",
        "depth_anything_colmap",
    ] = "depth_anything_colmap"
    camera_type: Literal["original", "refined"] = "refined"
    use_median_filter: bool = False
    num_targets_per_frame: int = 1
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    load_from_cache: bool = False
    skip_load_imgs: bool = False


class iPhoneDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        start: int = 0,
        end: int = -1,
        factor: int = 1,
        split: Literal["train", "val"] = "train",
        depth_type: Literal[
            "midas",
            "depth_anything",
            "lidar",
            "depth_anything_colmap",
        ] = "depth_anything_colmap",
        camera_type: Literal["original", "refined"] = "refined",
        use_median_filter: bool = False,
        num_targets_per_frame: int = 1,
        scene_norm_dict: SceneNormDict | None = None,
        load_from_cache: bool = False,
        skip_load_imgs: bool = False,
        **_,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.training = split == "train"
        self.split = split
        self.factor = factor
        self.start = start
        self.end = end
        self.depth_type = depth_type
        self.camera_type = camera_type
        self.use_median_filter = use_median_filter
        self.num_targets_per_frame = num_targets_per_frame
        self.scene_norm_dict = scene_norm_dict
        self.load_from_cache = load_from_cache
        self.cache_dir = osp.join(data_dir, "flow3d_preprocessed", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Test if the current data has validation set.
        with open(osp.join(data_dir, "splits", "val.json")) as f:
            split_dict = json.load(f)
        self.has_validation = len(split_dict["frame_names"]) > 0

        # Load metadata.
        with open(osp.join(data_dir, "splits", f"{split}.json")) as f:
            split_dict = json.load(f)
        full_len = len(split_dict["frame_names"])
        end = min(end, full_len) if end > 0 else full_len
        self.end = end
        self.frame_names = split_dict["frame_names"][start:end]
        time_ids = [t for t in split_dict["time_ids"] if t >= start and t < end]
        self.time_ids = torch.tensor(time_ids) - start
        guru.info(f"{self.time_ids.min()=} {self.time_ids.max()=}")
        # with open(osp.join(data_dir, "dataset.json")) as f:
        #     dataset_dict = json.load(f)
        # self.num_frames = dataset_dict["num_exemplars"]
        guru.info(f"{self.num_frames=}")
        with open(osp.join(data_dir, "extra.json")) as f:
            extra_dict = json.load(f)
        self.fps = float(extra_dict["fps"])

        # Load cameras.
        if self.camera_type == "original":
            Ks, w2cs = [], []
            for frame_name in self.frame_names:
                with open(osp.join(data_dir, "camera", f"{frame_name}.json")) as f:
                    camera_dict = json.load(f)
                focal_length = camera_dict["focal_length"]
                principal_point = camera_dict["principal_point"]
                Ks.append(
                    [
                        [focal_length, 0.0, principal_point[0]],
                        [0.0, focal_length, principal_point[1]],
                        [0.0, 0.0, 1.0],
                    ]
                )
                orientation = np.array(camera_dict["orientation"])
                position = np.array(camera_dict["position"])
                w2cs.append(
                    np.block(
                        [
                            [orientation, -orientation @ position[:, None]],
                            [np.zeros((1, 3)), np.ones((1, 1))],
                        ]
                    ).astype(np.float32)
                )
            self.Ks = torch.tensor(Ks)
            self.Ks[:, :2] /= factor
            self.w2cs = torch.from_numpy(np.array(w2cs))
        elif self.camera_type == "refined":
            Ks, w2cs = get_colmap_camera_params(
                osp.join(data_dir, "flow3d_preprocessed/colmap/sparse/"),
                [frame_name + ".png" for frame_name in self.frame_names],
            )
            self.Ks = torch.from_numpy(Ks[:, :3, :3].astype(np.float32))
            self.Ks[:, :2] /= factor
            self.w2cs = torch.from_numpy(w2cs.astype(np.float32))
        if not skip_load_imgs:
            # Load images.
            imgs = torch.from_numpy(
                np.array(
                    [
                        iio.imread(
                            osp.join(self.data_dir, f"rgb/{factor}x/{frame_name}.png")
                        )
                        for frame_name in tqdm(
                            self.frame_names,
                            desc=f"Loading {self.split} images",
                            leave=False,
                        )
                    ],
                )
            )
            self.imgs = imgs[..., :3] / 255.0
            self.valid_masks = imgs[..., 3] / 255.0
            # Load masks.
            self.masks = (
                torch.from_numpy(
                    np.array(
                        [
                            iio.imread(
                                osp.join(
                                    self.data_dir,
                                    "flow3d_preprocessed/track_anything/",
                                    f"{factor}x/{frame_name}.png",
                                )
                            )
                            for frame_name in tqdm(
                                self.frame_names,
                                desc=f"Loading {self.split} masks",
                                leave=False,
                            )
                        ],
                    )
                )
                / 255.0
            )
            if self.training:
                # Load depths.
                def load_depth(frame_name):
                    if self.depth_type == "lidar":
                        depth = np.load(
                            osp.join(
                                self.data_dir,
                                f"depth/{factor}x/{frame_name}.npy",
                            )
                        )[..., 0]
                    else:
                        depth = np.load(
                            osp.join(
                                self.data_dir,
                                f"flow3d_preprocessed/aligned_{self.depth_type}/",
                                f"{factor}x/{frame_name}.npy",
                            )
                        )
                        depth[depth < 1e-3] = 1e-3
                        depth = 1.0 / depth
                    return depth

                self.depths = torch.from_numpy(
                    np.array(
                        [
                            load_depth(frame_name)
                            for frame_name in tqdm(
                                self.frame_names,
                                desc=f"Loading {self.split} depths",
                                leave=False,
                            )
                        ],
                        np.float32,
                    )
                )
                max_depth_values_per_frame = self.depths.reshape(
                    self.num_frames, -1
                ).max(1)[0]
                max_depth_value = max_depth_values_per_frame.median() * 2.5
                print("max_depth_value", max_depth_value)
                self.depths = torch.clamp(self.depths, 0, max_depth_value)
                # Median filter depths.
                # NOTE(hangg): This operator is very expensive.
                if self.use_median_filter:
                    for i in tqdm(
                        range(self.num_frames), desc="Processing depths", leave=False
                    ):
                        depth = masked_median_blur(
                            self.depths[[i]].unsqueeze(1).to("cuda"),
                            (
                                self.masks[[i]]
                                * self.valid_masks[[i]]
                                * (self.depths[[i]] > 0)
                            )
                            .unsqueeze(1)
                            .to("cuda"),
                        )[0, 0].cpu()
                        self.depths[i] = depth * self.masks[i] + self.depths[i] * (
                            1 - self.masks[i]
                        )
                # Load the query pixels from 2D tracks.
                self.query_tracks_2d = [
                    torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                "flow3d_preprocessed/2d_tracks/",
                                f"{factor}x/{frame_name}_{frame_name}.npy",
                            )
                        ).astype(np.float32)
                    )
                    for frame_name in self.frame_names
                ]
                guru.info(
                    f"{len(self.query_tracks_2d)=} {self.query_tracks_2d[0].shape=}"
                )

                # Load sam features.
                # sam_feat_dir = osp.join(
                #     data_dir, f"flow3d_preprocessed/sam_features/{factor}x"
                # )
                # assert osp.exists(sam_feat_dir), f"SAM features not exist!"
                # sam_features, original_size, input_size = load_sam_features(
                #     sam_feat_dir, self.frame_names
                # )
                # guru.info(f"{sam_features.shape=} {original_size=} {input_size=}")
                # self.sam_features = sam_features
                # self.sam_original_size = original_size
                # self.sam_input_size = input_size
            else:
                # Load covisible masks.
                self.covisible_masks = (
                    torch.from_numpy(
                        np.array(
                            [
                                iio.imread(
                                    osp.join(
                                        self.data_dir,
                                        "flow3d_preprocessed/covisible/",
                                        f"{factor}x/{split}/{frame_name}.png",
                                    )
                                )
                                for frame_name in tqdm(
                                    self.frame_names,
                                    desc=f"Loading {self.split} covisible masks",
                                    leave=False,
                                )
                            ],
                        )
                    )
                    / 255.0
                )

        if self.scene_norm_dict is None:
            cached_scene_norm_dict_path = osp.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if osp.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                print("loading cached scene norm dict...")
                self.scene_norm_dict = torch.load(
                    osp.join(self.cache_dir, "scene_norm_dict.pth")
                )
            elif self.training:
                # Compute the scene scale and transform for normalization.
                # Normalize the scene based on the foreground 3D tracks.
                subsampled_tracks_3d = self.get_tracks_3d(
                    num_samples=10000, step=self.num_frames // 10, show_pbar=False
                )[0]
                scene_center = subsampled_tracks_3d.mean((0, 1))
                tracks_3d_centered = subsampled_tracks_3d - scene_center
                min_scale = tracks_3d_centered.quantile(0.05, dim=0)
                max_scale = tracks_3d_centered.quantile(0.95, dim=0)
                scale = torch.max(max_scale - min_scale).item() / 2.0
                original_up = -F.normalize(self.w2cs[:, 1, :3].mean(0), dim=-1)
                target_up = original_up.new_tensor([0.0, 0.0, 1.0])
                R = roma.rotvec_to_rotmat(
                    F.normalize(original_up.cross(target_up), dim=-1)
                    * original_up.dot(target_up).acos_()
                )
                transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
                self.scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
                torch.save(self.scene_norm_dict, cached_scene_norm_dict_path)
            else:
                raise ValueError("scene_norm_dict must be provided for validation.")

        # Normalize the scene.
        scale = self.scene_norm_dict["scale"]
        transfm = self.scene_norm_dict["transfm"]
        self.w2cs = self.w2cs @ torch.linalg.inv(transfm)
        self.w2cs[:, :3, 3] /= scale
        if self.training and not skip_load_imgs:
            self.depths /= scale

        if not skip_load_imgs:
            guru.info(
                f"{self.imgs.shape=} {self.valid_masks.shape=} {self.masks.shape=}"
            )

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def __len__(self):
        return self.imgs.shape[0]

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_img_wh(self) -> tuple[int, int]:
        return iio.imread(
            osp.join(self.data_dir, f"rgb/{self.factor}x/{self.frame_names[0]}.png")
        ).shape[1::-1]

    # def get_sam_features(self) -> list[torch.Tensor, tuple[int, int], tuple[int, int]]:
    #     return self.sam_features, self.sam_original_size, self.sam_input_size

    def get_tracks_3d(
        self, num_samples: int, step: int = 1, show_pbar: bool = True, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get 3D tracks from the dataset.

        Args:
            num_samples (int | None): The number of samples to fetch. If None,
                fetch all samples. If not None, fetch roughly a same number of
                samples across each frame. Note that this might result in
                number of samples less than what is specified.
            step (int): The step to temporally subsample the track.
        """
        assert (
            self.split == "train"
        ), "fetch_tracks_3d is only available for the training split."
        cached_track_3d_path = osp.join(self.cache_dir, f"tracks_3d_{num_samples}.pth")
        if osp.exists(cached_track_3d_path) and step == 1 and self.load_from_cache:
            print("loading cached 3d tracks data...")
            start, end = self.start, self.end
            cached_track_3d_data = torch.load(cached_track_3d_path)
            tracks_3d, visibles, invisibles, confidences, track_colors = (
                cached_track_3d_data["tracks_3d"][:, start:end],
                cached_track_3d_data["visibles"][:, start:end],
                cached_track_3d_data["invisibles"][:, start:end],
                cached_track_3d_data["confidences"][:, start:end],
                cached_track_3d_data["track_colors"],
            )
            return tracks_3d, visibles, invisibles, confidences, track_colors

        # Load 2D tracks.
        raw_tracks_2d = []
        candidate_frames = list(range(0, self.num_frames, step))
        num_sampled_frames = len(candidate_frames)
        for i in (
            tqdm(candidate_frames, desc="Loading 2D tracks", leave=False)
            if show_pbar
            else candidate_frames
        ):
            curr_num_samples = self.query_tracks_2d[i].shape[0]
            num_samples_per_frame = (
                int(np.floor(num_samples / num_sampled_frames))
                if i != candidate_frames[-1]
                else num_samples
                - (num_sampled_frames - 1)
                * int(np.floor(num_samples / num_sampled_frames))
            )
            if num_samples_per_frame < curr_num_samples:
                track_sels = np.random.choice(
                    curr_num_samples, (num_samples_per_frame,), replace=False
                )
            else:
                track_sels = np.arange(0, curr_num_samples)
            curr_tracks_2d = []
            for j in range(0, self.num_frames, step):
                if i == j:
                    target_tracks_2d = self.query_tracks_2d[i]
                else:
                    target_tracks_2d = torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                "flow3d_preprocessed/2d_tracks/",
                                f"{self.factor}x/"
                                f"{self.frame_names[i]}_"
                                f"{self.frame_names[j]}.npy",
                            )
                        ).astype(np.float32)
                    )
                curr_tracks_2d.append(target_tracks_2d[track_sels])
            raw_tracks_2d.append(torch.stack(curr_tracks_2d, dim=1))
        guru.info(f"{step=} {len(raw_tracks_2d)=} {raw_tracks_2d[0].shape=}")

        # Process 3D tracks.
        inv_Ks = torch.linalg.inv(self.Ks)[::step]
        c2ws = torch.linalg.inv(self.w2cs)[::step]
        H, W = self.imgs.shape[1:3]
        filtered_tracks_3d, filtered_visibles, filtered_track_colors = [], [], []
        filtered_invisibles, filtered_confidences = [], []
        masks = self.masks * self.valid_masks * (self.depths > 0)
        masks = (masks > 0.5).float()
        for i, tracks_2d in enumerate(raw_tracks_2d):
            tracks_2d = tracks_2d.swapdims(0, 1)
            tracks_2d, occs, dists = (
                tracks_2d[..., :2],
                tracks_2d[..., 2],
                tracks_2d[..., 3],
            )
            # visibles = postprocess_occlusions(occs, dists)
            visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
            # Unproject 2D tracks to 3D.
            track_depths = F.grid_sample(
                self.depths[::step, None],
                normalize_coords(tracks_2d[..., None, :], H, W),
                align_corners=True,
                padding_mode="border",
            )[:, 0]
            tracks_3d = (
                torch.einsum(
                    "nij,npj->npi",
                    inv_Ks,
                    F.pad(tracks_2d, (0, 1), value=1.0),
                )
                * track_depths
            )
            tracks_3d = torch.einsum(
                "nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0)
            )[..., :3]
            # Filter out out-of-mask tracks.
            is_in_masks = (
                F.grid_sample(
                    masks[::step, None],
                    normalize_coords(tracks_2d[..., None, :], H, W),
                    align_corners=True,
                ).squeeze()
                == 1
            )
            visibles *= is_in_masks
            invisibles *= is_in_masks
            confidences *= is_in_masks.float()
            # Get track's color from the query frame.
            track_colors = (
                F.grid_sample(
                    self.imgs[i * step : i * step + 1].permute(0, 3, 1, 2),
                    normalize_coords(tracks_2d[i : i + 1, None, :], H, W),
                    align_corners=True,
                    padding_mode="border",
                )
                .squeeze()
                .T
            )
            # at least visible 5% of the time, otherwise discard
            visible_counts = visibles.sum(0)
            valid = visible_counts >= min(
                int(0.05 * self.num_frames),
                visible_counts.float().quantile(0.1).item(),
            )

            filtered_tracks_3d.append(tracks_3d[:, valid])
            filtered_visibles.append(visibles[:, valid])
            filtered_invisibles.append(invisibles[:, valid])
            filtered_confidences.append(confidences[:, valid])
            filtered_track_colors.append(track_colors[valid])

        filtered_tracks_3d = torch.cat(filtered_tracks_3d, dim=1).swapdims(0, 1)
        filtered_visibles = torch.cat(filtered_visibles, dim=1).swapdims(0, 1)
        filtered_invisibles = torch.cat(filtered_invisibles, dim=1).swapdims(0, 1)
        filtered_confidences = torch.cat(filtered_confidences, dim=1).swapdims(0, 1)
        filtered_track_colors = torch.cat(filtered_track_colors, dim=0)
        if step == 1:
            torch.save(
                {
                    "tracks_3d": filtered_tracks_3d,
                    "visibles": filtered_visibles,
                    "invisibles": filtered_invisibles,
                    "confidences": filtered_confidences,
                    "track_colors": filtered_track_colors,
                },
                cached_track_3d_path,
            )
        return (
            filtered_tracks_3d,
            filtered_visibles,
            filtered_invisibles,
            filtered_confidences,
            filtered_track_colors,
        )

    def get_bkgd_points(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = self.imgs.shape[1:3]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(W, dtype=torch.float32),
                torch.arange(H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )
        candidate_frames = list(range(self.num_frames))
        num_sampled_frames = len(candidate_frames)
        bkgd_points, bkgd_point_normals, bkgd_point_colors = [], [], []
        for i in tqdm(candidate_frames, desc="Loading bkgd points", leave=False):
            img = self.imgs[i]
            depth = self.depths[i]
            bool_mask = ((1.0 - self.masks[i]) * self.valid_masks[i] * (depth > 0)).to(
                torch.bool
            )
            w2c = self.w2cs[i]
            K = self.Ks[i]
            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]
            curr_num_samples = points.shape[0]
            num_samples_per_frame = (
                int(np.floor(num_samples / num_sampled_frames))
                if i != candidate_frames[-1]
                else num_samples
                - (num_sampled_frames - 1)
                * int(np.floor(num_samples / num_sampled_frames))
            )
            if num_samples_per_frame < curr_num_samples:
                point_sels = np.random.choice(
                    curr_num_samples, (num_samples_per_frame,), replace=False
                )
            else:
                point_sels = np.arange(0, curr_num_samples)
            bkgd_points.append(points[point_sels])
            bkgd_point_normals.append(point_normals[point_sels])
            bkgd_point_colors.append(point_colors[point_sels])
        bkgd_points = torch.cat(bkgd_points, dim=0)
        bkgd_point_normals = torch.cat(bkgd_point_normals, dim=0)
        bkgd_point_colors = torch.cat(bkgd_point_colors, dim=0)
        return bkgd_points, bkgd_point_normals, bkgd_point_colors

    def get_video_dataset(self) -> Dataset:
        return iPhoneDatasetVideoView(self)

    def __getitem__(self, index: int):
        if self.training:
            index = np.random.randint(0, self.num_frames)
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": self.time_ids[index],
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.imgs[index],
            # (H, W).
            "valid_masks": self.valid_masks[index],
            # (H, W).
            "masks": self.masks[index],
        }
        if self.training:
            # (H, W).
            data["depths"] = self.depths[index]
            # (P, 2).
            data["query_tracks_2d"] = self.query_tracks_2d[index][:, :2]
            target_inds = torch.from_numpy(
                np.random.choice(
                    self.num_frames, (self.num_targets_per_frame,), replace=False
                )
            )
            # (N, P, 4).
            target_tracks_2d = torch.stack(
                [
                    torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                "flow3d_preprocessed/2d_tracks/",
                                f"{self.factor}x/"
                                f"{self.frame_names[index]}_"
                                f"{self.frame_names[target_index.item()]}.npy",
                            )
                        ).astype(np.float32)
                    )
                    for target_index in target_inds
                ],
                dim=0,
            )
            # (N,).
            target_ts = self.time_ids[target_inds]
            data["target_ts"] = target_ts
            # (N, 4, 4).
            data["target_w2cs"] = self.w2cs[target_ts]
            # (N, 3, 3).
            data["target_Ks"] = self.Ks[target_ts]
            # (N, P, 2).
            data["target_tracks_2d"] = target_tracks_2d[..., :2]
            # (N, P).
            (
                data["target_visibles"],
                data["target_invisibles"],
                data["target_confidences"],
            ) = parse_tapir_track_info(
                target_tracks_2d[..., 2], target_tracks_2d[..., 3]
            )
            # (N, P).
            data["target_track_depths"] = F.grid_sample(
                self.depths[target_inds, None],
                normalize_coords(
                    target_tracks_2d[..., None, :2],
                    self.imgs.shape[1],
                    self.imgs.shape[2],
                ),
                align_corners=True,
                padding_mode="border",
            )[:, 0, :, 0]
        else:
            # (H, W).
            data["covisible_masks"] = self.covisible_masks[index]
        return data

    def preprocess(self, data):
        return data


class iPhoneDatasetKeypointView(Dataset):
    """Return a dataset view of the annotated keypoints."""

    def __init__(self, dataset: iPhoneDataset):
        super().__init__()
        self.dataset = dataset
        assert self.dataset.split == "train"
        # Load 2D keypoints.
        keypoint_paths = sorted(
            glob(osp.join(self.dataset.data_dir, "keypoint/2x/train/0_*.json"))
        )
        keypoints = []
        for keypoint_path in keypoint_paths:
            with open(keypoint_path) as f:
                keypoints.append(json.load(f))
        time_ids = [
            int(osp.basename(p).split("_")[1].split(".")[0]) for p in keypoint_paths
        ]
        # only use time ids that are in the dataset.
        start = self.dataset.start
        time_ids = [t - start for t in time_ids if t - start in self.dataset.time_ids]
        self.time_ids = torch.tensor(time_ids)
        self.time_pairs = torch.tensor(list(product(self.time_ids, repeat=2)))
        self.index_pairs = torch.tensor(
            list(product(range(len(self.time_ids)), repeat=2))
        )
        self.keypoints = torch.tensor(keypoints, dtype=torch.float32)
        self.keypoints[..., :2] *= 2.0 / self.dataset.factor

    def __len__(self):
        return len(self.time_pairs)

    def __getitem__(self, index: int):
        ts = self.time_pairs[index]
        return {
            "ts": ts,
            "w2cs": self.dataset.w2cs[ts],
            "Ks": self.dataset.Ks[ts],
            "imgs": self.dataset.imgs[ts],
            "keypoints": self.keypoints[self.index_pairs[index]],
        }


class iPhoneDatasetVideoView(Dataset):
    """Return a dataset view of the video trajectory."""

    def __init__(self, dataset: iPhoneDataset):
        super().__init__()
        self.dataset = dataset
        self.fps = self.dataset.fps
        assert self.dataset.split == "train"

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, index):
        return {
            "frame_names": self.dataset.frame_names[index],
            "ts": index,
            "w2cs": self.dataset.w2cs[index],
            "Ks": self.dataset.Ks[index],
            "imgs": self.dataset.imgs[index],
            "depths": self.dataset.depths[index],
            "masks": self.dataset.masks[index],
        }


"""
class iPhoneDataModule(BaseDataModule[iPhoneDataset]):
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        start: int = 0,
        end: int = -1,
        depth_type: Literal[
            "midas",
            "depth_anything",
            "lidar",
            "depth_anything_colmap",
        ] = "depth_anything_colmap",
        camera_type: Literal["original", "refined"] = "refined",
        use_median_filter: bool = False,
        num_targets_per_frame: int = 1,
        load_from_cache: bool = False,
        **kwargs,
    ):
        super().__init__(dataset_cls=iPhoneDataset, **kwargs)
        self.data_dir = data_dir
        self.start = start
        self.end = end
        self.factor = factor
        self.depth_type = depth_type
        self.camera_type = camera_type
        self.use_median_filter = use_median_filter
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache

        self.val_loader_tasks = ["img", "keypoint"]

    def setup(self, *_, **__) -> None:
        guru.info("Loading train dataset...")
        self.train_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            training=True,
            split="train",
            start=self.start,
            end=self.end,
            factor=self.factor,
            depth_type=self.depth_type,  # type: ignore
            camera_type=self.camera_type,  # type: ignore
            use_median_filter=self.use_median_filter,
            num_targets_per_frame=self.num_targets_per_frame,
            max_steps=self.max_steps * self.batch_size,
            load_from_cache=self.load_from_cache,
        )
        if self.train_dataset.has_validation:
            guru.info("Loading val dataset...")
            self.val_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                training=False,
                split="val",
                start=self.start,
                end=self.end,
                factor=self.factor,
                depth_type=self.depth_type,  # type: ignore
                camera_type=self.camera_type,  # type: ignore
                use_median_filter=self.use_median_filter,
                scene_norm_dict=self.train_dataset.scene_norm_dict,
                load_from_cache=self.load_from_cache,
            )
        else:
            # Dummy validation set.
            self.val_dataset = TensorDataset(torch.zeros(0))  # type: ignore
        self.keypoint_dataset = iPhoneDatasetKeypointView(self.train_dataset)
        self.video_dataset = self.train_dataset.get_video_dataset()
        guru.success("Loading finished!")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=iPhoneDataset.train_collate_fn,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [DataLoader(self.val_dataset), DataLoader(self.keypoint_dataset)]
        """
