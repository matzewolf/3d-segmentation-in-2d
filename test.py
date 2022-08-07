import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from dataset import ShapeNetPartDataset
from model import MultiScaleUNet


def test(s_test, p_test, x_test, y_test,
         model, class_label_region,
         test_dataset, result_path, device):
    """
    Function to predict and store the predicted values in a .hdf5 file
    :param s_test: point cloud segments
    :param p_test: 2D position
    :param x_test: point cloud position in 3D
    :param y_test: point cloud shape class
    :param model: pre-trained model
    :param class_label_region: class part label intervals
    :param test_dataset: ShapeNetPartDataset 'test' split
    :param result_path: directory to store the .hdf5 file

    :return pre_test: predicted labels
    """
    test_set_len = len(y_test)
    with h5py.File(result_path, 'w') as f:
        f.create_dataset('x_test', data=x_test)  # p cloud position in 3D
        f.create_dataset('y_test', data=y_test)  # p cloud shape class
        f.create_dataset('s_test', data=s_test)  # p cloud segments
        f.create_dataset('p_test', data=p_test)  # 2D position
        pre_set = f.create_dataset('pre_test', shape=(test_set_len, 2048, 1),
                                   dtype=np.int64)
        pre_test = np.zeros_like(s_test)

        for idx_sample, pos, obj_class in zip(tqdm(range(0, len(p_test)),
                                                   desc='point segments'),
                                              p_test, y_test):
            input_tensor = test_dataset.__getitem__(idx_sample)["3d_points"]
            input_tensor = input_tensor[None, :]
            input_tensor = input_tensor.contiguous().to(device)
            pre_image = model(input_tensor).to('cpu')
            label_min = int(class_label_region[obj_class, 0])
            label_max = int(class_label_region[obj_class, 1] + 1)
            pre_image = pre_image[:, label_min:label_max, :, :] \
                .argmax(1) + label_min
            pre_sample = pre_image[:, pos[:, 0], pos[:, 1]]

            pre_test[idx_sample] = pre_sample[:, None][0]
            pre_set[idx_sample] = pre_sample.T
    return pre_test


def evaluate(model_path):
    """
    Evaluates the model.
    :param model_path: path to trained model
    """
    # reading class names of the ShapeNet dataset
    class_name = np.genfromtxt('all_object_categories.txt', dtype='U')[:, 0]

    # reading the test dataset from disk
    dataset_path = Path("shapenet_prepared.h5")
    with h5py.File(dataset_path, 'r') as f:
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]
        s_test = f['s_test'][:]
        p_test = f['p_test'][:]

    # creating class part label intervals
    class_label_region = np.zeros((16, 2), dtype=np.int64)
    for i_class in range(16):
        idx_list = np.where(y_test == i_class)[0]
        gt_list = s_test[idx_list]
        label_min = gt_list.min()
        label_max = gt_list.max()
        class_label_region[i_class, 0] = label_min
        class_label_region[i_class, 1] = label_max

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    # reading the trained model
    model = MultiScaleUNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    # creating test split
    test_dataset = ShapeNetPartDataset(split='test')

    # creating new file to store results
    result_path = Path('ShapeNet_testing_result.hdf5')
    pre_test = test(s_test, p_test, x_test, y_test, model,
                    class_label_region, test_dataset, result_path, device)

    # calculate iou for each shape
    iou_shape = np.zeros(len(s_test))
    for idx_sample, pre_sample, gt_sample, obj_class in \
            zip(tqdm(range(len(s_test)), desc='iou calculation'),
                pre_test, s_test, y_test):
        label_min = int(class_label_region[obj_class, 0])
        label_max = int(class_label_region[obj_class, 1] + 1)
        iou_list = []
        # for each segment, calculate iou
        for i_class in range(label_min, label_max):
            tp = np.sum((pre_sample == i_class) * (gt_sample == i_class))
            fp = np.sum((pre_sample == i_class) * (gt_sample != i_class))
            fn = np.sum((pre_sample != i_class) * (gt_sample == i_class))
            # if current segment exists then count the iou
            iou = (tp+1e-12) / (tp+fp+fn+1e-12)
            iou_list.append(iou)
        iou_shape[idx_sample] = np.mean(iou_list)
    print(f'iou_instance = {iou_shape.mean()}')

    iou_class = np.zeros(16)
    for obj_class in range(16):
        iou_obj_class = iou_shape[y_test[:] == obj_class]
        iou_class[obj_class] = iou_obj_class.mean()
    print(f'iou_class = {iou_class.mean()}')
    for obj_class in range(16):
        print(f'class {obj_class}, {class_name[obj_class]}: ', end='')
        print(f'iou = {iou_class[obj_class]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument("--model_path", default="./model_best.ckpt", type=str)
    args = parser.parse_args()
    evaluate(args.model_path)
