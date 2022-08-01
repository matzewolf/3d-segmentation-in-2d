from pathlib import Path

import h5py
import numpy as np

from utils import gpgl2_segmentation
from utils import jitter_point_cloud


def combine_split(split, files, result_file):
    """
    Combines all HDF5 files for a specific data split into a new combined HDF5
    file.

    :param split: Which data split to process, either 'train', 'val' or 'test'.
    :param files: List of file paths of HDF5 files that belong to this split.
    :param result_file: HDF5 file where the results should be written into.
    """
    assert split in ['train', 'val', 'test'], ValueError("Invalid split")

    split_length = 0
    for file in files:
        with h5py.File(file, 'r') as f:
            split_length += len(f['label'])
    print(f"{split} set length: {split_length}")

    points_3d = result_file.create_dataset(
        f"x_{split}",
        (split_length, 2048, 3),
        dtype='<f4'
    )
    labels = result_file.create_dataset(
        f"y_{split}",
        (split_length, 1),
        dtype='|u1'
    )
    parts = result_file.create_dataset(
        f"s_{split}",
        (split_length, 2048),
        dtype='|u1'
    )

    offset = 0
    for file in files:
        with h5py.File(file, 'r') as f:
            print(f"combining dataset: {file.name}")
            dataset_length = len(f['label'])
            points_3d[offset:offset + dataset_length] = f['data']
            labels[offset:offset + dataset_length] = f['label']
            parts[offset:offset + dataset_length] = f['pid']
            offset += dataset_length


def combine_dataset(data_path, combined_hdf5_file):
    """
    Combines all raw HDF5 data files for the ShapeNet Part dataset into one
    combined HDF5 data file.

    :param data_path: Path to the downloaded ShapeNet Part directory.
    :param combined_hdf5_file: Path to the generated combined HDF5 file.
    """
    train_files = []
    val_files = []
    test_files = []
    for file in sorted(data_path.iterdir()):
        if file.name.startswith("ply_data_train"):
            train_files.append(file)
        elif file.name.startswith("ply_data_val"):
            val_files.append(file)
        elif file.name.startswith("ply_data_test"):
            test_files.append(file)

    with h5py.File(combined_hdf5_file, 'w') as result_file:
        combine_split('train', train_files, result_file)
        combine_split('val', val_files, result_file)
        combine_split('test', test_files, result_file)


def prepare_split(split, input_file, result_file,
                  num_repeats_train = 5,
                  jitter_train_points= True):
    """
    Prepares each split such that 3D point clouds with part annotations are
    transformed into 2D images where pixels are assigned to 3D points with part
    annotations. See the paper for more details.

    :param split: Which data split to process, either 'train', 'val' or 'test'.
    :param input_file: Path to the combined ShapeNet Part HDF5 data file.
    :param result_file: HDF5 file where the results should be written into.
    :param num_repeats_train: How many times the training data should be
        repeatedly processed.
    :param jitter_train_points: Weather the training data points should be
        jittered with some noise.
    """
    assert split in ['train', 'val', 'test'], ValueError("Invalid split")

    if split == 'train':
        num_repeats = num_repeats_train
    else:
        num_repeats = 1

    with h5py.File(input_file, 'r') as f:
        num_instances, num_points = f[f"s_{split}"].shape
        points_3d = f[f"x_{split}"][:]
        labels = f[f"y_{split}"][:]
        parts = f[f"s_{split}"][:]

    result_points_3d = result_file.create_dataset(
        f"x_{split}",
        (num_instances * num_repeats, num_points, 3),
        dtype='f'
    )
    result_labels = result_file.create_dataset(
        f"y_{split}",
        (num_instances * num_repeats),
        dtype='i'
    )
    result_parts = result_file.create_dataset(
        f"s_{split}",
        (num_instances * num_repeats, num_points),
        dtype='i'
    )
    result_points_2d = result_file.create_dataset(
        f"p_{split}",
        (num_instances * num_repeats, num_points, 2),
        dtype='i'
    )

    sample_idx = 0
    node_losses = []
    for repeat_idx in range(num_repeats):
        for instance_idx in range(num_instances):
            current_points_3d = points_3d[instance_idx]
            current_label = labels[instance_idx]
            current_parts = parts[instance_idx]
            current_points_3d = current_points_3d[np.newaxis, :, :]
            if jitter_train_points and split == 'train':
                current_points_3d = jitter_point_cloud(current_points_3d)
            current_points_2d, node_loss = gpgl2_segmentation(
                current_points_3d[0])
            result_points_3d[sample_idx] = current_points_3d
            result_labels[sample_idx] = current_label
            result_parts[sample_idx] = current_parts
            result_points_2d[sample_idx] = current_points_2d
            sample_idx += 1
            node_losses.append(node_loss)
            print(f"Split {split}: iteration {sample_idx}, loss {node_loss}")

    mean_node_loss = np.array(node_losses).mean()
    result_file[f"num_repeats_{split}"] = num_repeats
    result_file[f"mean node loss_{split}"] = mean_node_loss
    print(f"{split} loss: {mean_node_loss}")


def prepare_dataset(combined_hdf5_file, prepared_hdf5_file):
    """
    Prepares the combined HDF5 data with 3D point clouds with part annotations
    into 2D images where pixels are assigned to 3D points with part
    annotations. See the paper for more details.

    :param combined_hdf5_file: Path to the combined HDF5 data file.
    :param prepared_hdf5_file: Path to the prepared HDF5 data file.
    """
    with h5py.File(prepared_hdf5_file, 'w') as result_file:
        prepare_split('train', combined_hdf5_file, result_file)
        prepare_split('val', combined_hdf5_file, result_file)
        prepare_split('test', combined_hdf5_file, result_file)


def main():
    """
    Script that executes both the data combination and preparation step.
    """
    data_path = Path('hdf5_data')
    combined_hdf5_file = Path('shapenet_combined.h5')
    prepared_hdf5_file = Path('shapenet_prepared.h5')
    combine_dataset(data_path, combined_hdf5_file)
    prepare_dataset(combined_hdf5_file, prepared_hdf5_file)


if __name__ == '__main__':
    main()
