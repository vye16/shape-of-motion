from abc import abstractmethod

import torch
from torch.utils.data import Dataset, default_collate


class BaseDataset(Dataset):
    @property
    @abstractmethod
    def num_frames(self) -> int: ...

    @abstractmethod
    def get_w2cs(self) -> torch.Tensor: ...

    @abstractmethod
    def get_Ks(self) -> torch.Tensor: ...

    @abstractmethod
    def get_img_wh(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_tracks_3d(
        self, num_samples: int, **kwargs
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]: ...

    @abstractmethod
    def get_bkgd_points(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @staticmethod
    def train_collate_fn(batch):
        collated = {}
        for k in batch[0]:
            if k not in [
                "query_tracks_2d",
                "target_ts",
                "target_w2cs",
                "target_Ks",
                "target_tracks_2d",
                "target_visibles",
                "target_track_depths",
                "target_invisibles",
                "target_confidences",
            ]:
                collated[k] = default_collate([sample[k] for sample in batch])
            else:
                collated[k] = [sample[k] for sample in batch]
        return collated
