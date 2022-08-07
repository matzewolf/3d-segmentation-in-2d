from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from prepare_dataset import combine_dataset
from prepare_dataset import prepare_dataset


class ShapeNetPartDataset(Dataset):
    """
    PyTorch dataset class for the Stanford ShapeNet Part dataset, however with
    some specific preprocessing steps, see `combine_dataset` and
    `prepare_dataset` as well as the paper.

    :param raw_data_path: Path to the downloaded ShapeNet Part directory.
    :param combined_data_file: Path to the HDF5 file for the combined data.
    :param prepared_data_file: Path to the HDF5 file for the prepared data.
    :param split: Which data split to process, either 'train', 'val', 'test'
        or 'overfit'. Overfit dataset consists of the first instance in the
        val dataset and can be used for sanity checks on the train/val
        process.
    :param size_grid: Grid size of the image
    """
    def __init__(self, raw_data_path: Path = Path('hdf5_data'),
                 combined_data_file: Path = Path('shapenet_combined.h5'),
                 prepared_data_file: Path = Path('shapenet_prepared.h5'),
                 split: str = 'train', size_grid: int = 16):
        self.num_classes = 50
        self.size_image = size_grid ** 2
        if not prepared_data_file.exists():
            if not combined_data_file.exists():
                combine_dataset(raw_data_path, combined_data_file)
            prepare_dataset(combined_data_file, prepared_data_file)
        if split not in ['train', 'val', 'test', 'overfit']:
            raise ValueError("Invalid split")
        self.data = h5py.File(prepared_data_file, 'r')
        data_split = 'val' if split == 'overfit' else split
        self.points_3d = torch.tensor(np.array(self.data[f"x_{data_split}"]),
                                      dtype=torch.float32)
        self.parts = torch.tensor(np.array(self.data[f"s_{data_split}"]),
                                  dtype=torch.int64)
        self.points_2d = torch.tensor(np.array(self.data[f"p_{data_split}"]),
                                      dtype=torch.int64)
        if split == 'overfit':
            self.points_3d = torch.unsqueeze(self.points_3d[0], 0)
            self.parts = torch.unsqueeze(self.parts[0], 0)
            self.points_2d = torch.unsqueeze(self.points_2d[0], 0)
        assert len(self.points_3d) == len(self.parts) == len(self.points_2d), \
            "Instance dimension of 3D points, parts and 2D points do not match"

    def __getitem__(self, item) -> dict:
        points_2d_item = self.points_2d[item]
        points_3d_image = torch.zeros(self.size_image, self.size_image, 3,
                                      dtype=torch.float32)
        parts_image = torch.zeros(self.size_image, self.size_image,
                                  dtype=torch.int64) + self.num_classes
        points_3d_image[points_2d_item[:, 0],
                        points_2d_item[:, 1]] = self.points_3d[item]
        parts_image[points_2d_item[:, 0],
                    points_2d_item[:, 1]] = self.parts[item]
        return {
            "3d_points": torch.permute(points_3d_image,
                                       (2, 0, 1)),  # adjust dim to [N,C,H,W]
            "part_label": parts_image
        }

    def __len__(self):
        return len(self.points_3d)

    @staticmethod
    def move_batch_to_device(batch, device):
        batch["3d_points"] = batch["3d_points"].to(device)
        batch["part_label"] = batch["part_label"].to(device)
