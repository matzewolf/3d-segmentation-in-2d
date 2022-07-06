from pathlib import Path

import h5py
import numpy as np

from utils import jitter_point_cloud, gpgl2_segmentation


def combine_split(split: str, files: list[Path], result_file: h5py.File):
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


def combine_dataset(data_path: Path, combined_hdf5_file: Path):
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


def prepare_split(split: str, input_file: Path, result_file: h5py.File, num_repeats_train: int = 5,
                  jitter_train_points: bool = True):
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
            current_points_2d, node_loss = gpgl2_segmentation(current_points_3d[0])
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


def prepare_dataset(combined_hdf5_file: Path, prepared_hdf5_file: Path):
    with h5py.File(prepared_hdf5_file, 'w') as result_file:
        prepare_split('train', combined_hdf5_file, result_file)
        prepare_split('val', combined_hdf5_file, result_file)
        prepare_split('test', combined_hdf5_file, result_file)


def main():
    data_path = Path('hdf5_data')
    combined_hdf5_file = Path('shapenet_combined.h5')
    prepared_hdf5_file = Path('shapenet_prepared.h5')
    combine_dataset(data_path, combined_hdf5_file)
    prepare_dataset(combined_hdf5_file, prepared_hdf5_file)


if __name__ == '__main__':
    main()
