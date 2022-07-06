from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from prepare_dataset import combine_dataset, prepare_dataset


class ShapeNetPartDataset(Dataset):
    def __init__(self, path: Path = Path('hdf5_data'),
                 split: str = 'train', size_sub: int = 16):
        self.num_classes = 50
        self.size_image = size_sub ** 2
        combined_hdf5_file = Path('shapenet_combined.h5')
        prepared_hdf5_file = Path('shapenet_prepared.h5')
        if not combined_hdf5_file.exists():
            combine_dataset(path, combined_hdf5_file)
        if not prepared_hdf5_file.exists():
            prepare_dataset(combined_hdf5_file, prepared_hdf5_file)

        assert split in ['train', 'val', 'test', 'overfit'], ValueError("Invalid split")
        self.data = h5py.File(path, 'r')
        self.points_3d = torch.from_numpy(self.data[f"x_{split}"])
        self.parts = torch.from_numpy(self.data[f"s_{split}"])
        self.points_2d = torch.from_numpy(self.data[f"p_{split}"])
        if split == 'overfit':
            self.points_3d = torch.unsqueeze(self.points_3d[0], 0)
            self.parts = torch.unsqueeze(self.parts[0], 0)
            self.points_2d = torch.unsqueeze(self.parts[0], 0)
        assert len(self.points_3d) == len(self.parts) == len(self.points_2d), \
            "Instance dimension of 3D points, parts and 2D points do not match"

    def __getitem__(self, item):
        points_2d_item = self.points_2d[item]
        points_3d_image = torch.zeros((self.size_image, self.size_image, 3),
                                      dtype=torch.float32)
        parts_image = torch.zeros((self.size_image, self.size_image, 3),
                                  dtype=torch.float32) + self.num_classes
        points_3d_image[points_2d_item[:, 0], points_2d_item[:, 1]] = self.points_3d[item]
        parts_image[points_2d_item[:, 0], points_2d_item[:, 1]] = self.parts[item]

    def __len__(self):
        return len(self.points_3d)
