from pathlib import Path

import h5py


def prepare_split(split: str, files: list[Path], result_file: h5py.File):
    assert split in ['train', 'val', 'test'], ValueError("Invalid split")

    split_length = 0
    for file in files:
        with h5py.File(file, 'r') as f:
            split_length += len(f['label'])
    print(f"{split} set length: {split_length}")

    points = result_file.create_dataset(f"x_{split}", (split_length, 2048, 3), dtype='<f4')
    label = result_file.create_dataset(f"y_{split}", (split_length, 1), dtype='|u1')
    part = result_file.create_dataset(f"s_{split}", (split_length, 2048), dtype='|u1')

    offset = 0
    for file in files:
        with h5py.File(file, 'r') as f:
            print(f"combining dataset: {file.name}")
            dataset_length = len(f['label'])
            points[offset:offset + dataset_length] = f['data']
            label[offset:offset + dataset_length] = f['label']
            part[offset:offset + dataset_length] = f['pid']
            offset += dataset_length


def prepare_dataset():
    data_path = Path('hdf5_data')
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

    with h5py.File('shapenet_combined.h5', 'w') as result_file:
        prepare_split('train', train_files, result_file)
        prepare_split('val', val_files, result_file)
        prepare_split('test', test_files, result_file)


if __name__ == '__main__':
    prepare_dataset()
