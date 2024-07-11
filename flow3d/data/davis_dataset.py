import os
from dataclasses import dataclass
from functools import partial
from typing import Literal

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from roma import roma
from tqdm import tqdm

from flow3d.data.base_dataset import BaseDataset
from flow3d.data.utils import (
    SceneNormDict,
    get_tracks_3d_for_query_frame,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4


@dataclass
class DavisDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = "480p"
    depth_type: Literal["depth_anything", "unidepth"] = "unidepth"
    camera_type: Literal["droid_recon"] = "droid_recon"
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 1
    load_from_cache: bool = False


class DavisDataset(BaseDataset):
    def __init__(
        self,
        seq_name: str,
        root_dir: str,
        start: int = 0,
        end: int = -1,
        res: str = "480p",
        depth_type: Literal["depth_anything", "unidepth"] = "unidepth",
        camera_type: Literal["droid_recon"] = "droid_recon",
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 1,
        load_from_cache: bool = False,
        **_,
    ):
        super().__init__()

        self.seq_name = seq_name
        self.root_dir = root_dir
        self.res = res
        self.depth_type = depth_type
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False

        self.img_dir = f"{root_dir}/JPEGImages/{res}/{seq_name}"
        self.depth_dir = f"{root_dir}/{depth_type}/{res}/{seq_name}"
        self.mask_dir = f"{root_dir}/Annotations/{res}/{seq_name}"
        self.tracks_dir = f"{root_dir}/2d_tracks/{res}/{seq_name}"
        self.cache_dir = f"{root_dir}/flow3d_preprocessed/{res}/{seq_name}"
        frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]

        if end == -1:
            end = len(frame_names)
        self.start = start
        self.end = end
        self.frame_names = frame_names[start:end]

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]

        # load cameras
        if camera_type == "droid_recon":
            img = self.get_image(0)
            H, W = img.shape[:2]
            w2cs, Ks = load_cameras(
                f"{root_dir}/{camera_type}/{res}/{seq_name}.npy", H, W
            )
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        assert (
            len(frame_names) == len(w2cs) == len(Ks)
        ), f"{len(frame_names)}, {len(w2cs)}, {len(Ks)}"
        self.w2cs = w2cs[start:end]
        self.Ks = Ks[start:end]

        if scene_norm_dict is None:
            cached_scene_norm_dict_path = os.path.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if os.path.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                print("loading cached scene norm dict...")
                scene_norm_dict = torch.load(
                    os.path.join(self.cache_dir, "scene_norm_dict.pth")
                )
            else:
                tracks_3d = self.get_tracks_3d(5000, step=self.num_frames // 10)[0]
                scene_norm_dict = compute_scene_norm(tracks_3d, self.w2cs)
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(scene_norm_dict, cached_scene_norm_dict_path)
        self.scene_norm_dict = scene_norm_dict

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def __len__(self):
        return len(self.frame_names)

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_img_wh(self) -> tuple[int, int]:
        return self.get_image(0).shape[1::-1]

    def get_image(self, index):
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        return self.imgs[index]

    def get_mask(self, index):
        if self.masks[index] is None:
            self.masks[index] = self.load_mask(index)
        return self.masks[index]

    def get_depth(self, index):
        if self.depths[index] is None:
            self.depths[index] = self.load_depth(index)
        return self.depths[index]

    def load_image(self, index):
        path = f"{self.img_dir}/{self.frame_names[index]}.jpg"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0

    def load_mask(self, index):
        path = f"{self.mask_dir}/{self.frame_names[index]}.png"
        mask = imageio.imread(path)
        mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        return torch.from_numpy(mask).float()

    def load_depth(self, index):
        path = f"{self.depth_dir}/{self.frame_names[index]}.png"
        depth = imageio.imread(path)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 256.0
        return torch.from_numpy(depth).float()

    def load_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1
    ):
        """
        tracks are 2d, occs and uncertainties
        :param dim (int): dimension to stack the time axis
        return (T, N, 4) if dim=0, (N, T, 4) if dim=1
        """
        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))

    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, step))
        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
        depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
        inv_Ks = torch.linalg.inv(self.Ks[target_idcs])
        c2ws = torch.linalg.inv(self.w2cs[target_idcs])

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0
        tracks_all_queries = []
        for q_idx in query_idcs:
            # (N, T, 4)
            tracks_2d = self.load_target_tracks(q_idx, target_idcs)
            num_sel = int(
                min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
            )
            if num_sel < len(tracks_2d):
                sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                tracks_2d = tracks_2d[sel_idcs]
            cur_num += tracks_2d.shape[0]
            img = self.get_image(q_idx)
            tracks_tuple = get_tracks_3d_for_query_frame(
                q_idx, img, tracks_2d, depths, masks, inv_Ks, c2ws
            )
            tracks_all_queries.append(tracks_tuple)
        tracks_3d, colors, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )
        return tracks_3d, visibles, invisibles, confidences, colors

    def get_bkgd_points(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(W, dtype=torch.float32),
                torch.arange(H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )
        query_idcs = list(range(start, end, step))
        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0
        bg = []
        for query_idx in tqdm(query_idcs, desc="Loading bkgd points", leave=False):
            img = self.get_image(query_idx)
            depth = self.get_depth(query_idx)
            mask = self.get_mask(query_idx)
            bool_mask = ((1.0 - mask) * (depth > 0)).to(torch.bool)
            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]
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
            num_sel = int(min(num_per_query_frame, num_samples - cur_num, len(points)))
            if num_sel < len(points):
                sel_idcs = np.random.choice(len(points), num_sel, replace=False)
                points = points[sel_idcs]
                point_normals = point_normals[sel_idcs]
                point_colors = point_colors[sel_idcs]
            cur_num += len(points)
            bg.append((points, point_normals, point_colors))
        bg_points, bg_normals, bg_colors = map(partial(torch.cat, dim=0), zip(*bg))
        return bg_points, bg_normals, bg_colors

    def __getitem__(self, index: int):
        index = np.random.randint(0, self.num_frames)
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": torch.tensor(index),
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.get_image(index),
            # (H, W).
            "masks": self.get_mask(index),
            "depths": self.get_depth(index),
        }
        # (P, 2)
        query_tracks = self.load_target_tracks(index, [index])[:, 0, :2]
        target_inds = torch.from_numpy(
            np.random.choice(
                self.num_frames, (self.num_targets_per_frame,), replace=False
            )
        )
        # (N, P, 4)
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), dim=0)
        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[target_inds]
        data["target_Ks"] = self.Ks[target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]
        # (N, P).
        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
        # (N, H, W)
        target_depths = torch.stack([self.get_depth(i) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0]
        return data


def load_cameras(path: str, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    recon = np.load(path, allow_pickle=True).item()
    print(f"{recon.keys()=}")
    traj_c2w = recon["traj_c2w"]  # (N, 4, 4)
    h, w = recon["img_shape"]
    sy, sx = H / h, W / w
    print(f"{sy=}, {sx=}")
    traj_w2c = np.linalg.inv(traj_c2w)
    fx, fy, cx, cy = recon["intrinsics"]  # (4,)
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
    Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)
    return torch.from_numpy(traj_w2c).float(), torch.from_numpy(Ks).float()


def compute_scene_norm(X: torch.Tensor, w2cs: torch.Tensor) -> SceneNormDict:
    """
    :param X: [N*T, 3]
    :param w2cs: [N, 4, 4]
    """
    X = X.reshape(-1, 3)
    scene_center = X.mean(dim=0)
    X = X - scene_center[None]
    min_scale = X.quantile(0.05, dim=0)
    max_scale = X.quantile(0.95, dim=0)
    scale = (max_scale - min_scale).max().item() / 2.0
    original_up = -F.normalize(w2cs[:, 1, :3].mean(0), dim=-1)
    target_up = original_up.new_tensor([0.0, 0.0, 1.0])
    R = roma.rotvec_to_rotmat(
        F.normalize(original_up.cross(target_up), dim=-1)
        * original_up.dot(target_up).acos_()
    )
    transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
    return SceneNormDict(scale=scale, transfm=transfm)


if __name__ == "__main__":
    d = DavisDataset("bear", "/shared/vye/datasets/DAVIS", camera_type="droid_recon")
