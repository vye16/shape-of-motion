from dataclasses import asdict, replace
from typing import Union

from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .davis_dataset import DavisDataConfig, DavisDataset
from .iphone_dataset import (
    iPhoneDataConfig,
    iPhoneDataset,
    iPhoneDatasetKeypointView,
    iPhoneDatasetVideoView,
)


def get_train_val_datasets(
    data_cfg: Union[iPhoneDataConfig, DavisDataConfig]
) -> tuple[BaseDataset, Dataset | None, Dataset | None, Dataset | None]:
    if isinstance(data_cfg, iPhoneDataConfig):
        train_dataset = iPhoneDataset(**asdict(data_cfg))
        train_video_view = iPhoneDatasetVideoView(train_dataset)
        val_img_dataset = (
            iPhoneDataset(
                **asdict(replace(data_cfg, split="val", load_from_cache=True))
            )
            if train_dataset.has_validation
            else None
        )
        val_kpt_dataset = iPhoneDatasetKeypointView(train_dataset)
    elif isinstance(data_cfg, DavisDataConfig):
        train_dataset = DavisDataset(**asdict(data_cfg))
        train_video_view = None
        val_img_dataset = None
        val_kpt_dataset = None
    else:
        raise ValueError(f"Unknown data config: {data_cfg}")
    return train_dataset, train_video_view, val_img_dataset, val_kpt_dataset
