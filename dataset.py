from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from prepare_dataset import combine_dataset, prepare_dataset


class ShapeNetPartDataset(Dataset):
    """
    PyTorch dataset class for the Stanford ShapeNet Part dataset, however with
    some specific preprocessing steps, see `combine_dataset` and
    `prepare_dataset` as well as the paper.

    :param path: Path to the downloaded ShapeNet Part directory.
    :param split: Which data split to process, either 'train', 'val', 'test'
        or 'overfit'. Overfit dataset consists of the first instance in the
        val dataset and can be used for sanity checks on the train/val
        process.
    :param size_sub: tdb
    """
    def __init__(self, path: Path = Path('hdf5_data'),
                 split: str = 'train', size_sub: int = 16):
        self.num_classes = 50
        self.size_image = size_sub ** 2
        combined_hdf5_file = Path('shapenet_combined.h5')
        prepared_hdf5_file = Path('shapenet_prepared.h5')
        if not prepared_hdf5_file.exists():
            if not combined_hdf5_file.exists():
                print("Combining the datasets...")
                combine_dataset(path, combined_hdf5_file)
            print("Preparing the datasets (this may take several hours)...")
            prepare_dataset(combined_hdf5_file, prepared_hdf5_file)
        assert split in ['train', 'val', 'test', 'overfit'], \
            ValueError("Invalid split")
        self.data = h5py.File(prepared_hdf5_file, 'r')
        if split != 'overfit':
            self.points_3d = torch.tensor(np.array(self.data[f"x_{split}"]))
            self.parts = torch.tensor(np.array(self.data[f"s_{split}"]))
            self.points_2d = torch.tensor(np.array(self.data[f"p_{split}"]))
        else:
            points_3d = torch.tensor(np.array(self.data["x_val"]))
            parts = torch.tensor(np.array(self.data["s_val"]))
            points_2d = torch.tensor(np.array(self.data["p_val"]))
            self.points_3d = torch.unsqueeze(points_3d[0], 0)
            self.parts = torch.unsqueeze(parts[0], 0)
            self.points_2d = torch.unsqueeze(points_2d[0], 0)
        assert len(self.points_3d) == len(self.parts) == len(self.points_2d), \
            "Instance dimension of 3D points, parts and 2D points do not match"

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        points_2d_item = self.points_2d[item].to(torch.long)
        points_3d_image = torch.zeros((self.size_image, self.size_image, 3),
                                      dtype=torch.float32)
        parts_image = torch.zeros((self.size_image, self.size_image),
                                  dtype=torch.int32) + self.num_classes
        points_3d_image[points_2d_item[:, 0],
                        points_2d_item[:, 1]] = self.points_3d[item]
        parts_image[points_2d_item[:, 0],
                    points_2d_item[:, 1]] = self.parts[item]
        return points_3d_image, parts_image

    def __len__(self) -> int:
        return len(self.points_3d)
