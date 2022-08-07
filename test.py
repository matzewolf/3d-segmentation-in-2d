import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from dataset import ShapeNetPartDataset
from model import MultiScaleUNet


def test(s_test, p_test, x_test, y_test,
         model, class_label_region,
         test_dataset, result_path):
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

        for idx_sample, pos, obj_class in zip(range(0, len(p_test)),
                                              p_test, y_test):
            input_tensor = torch.tensor(test_dataset.
                                        __getitem__(idx_sample)["3d_points"])
            input_tensor = input_tensor[None, :]
            pre_image = model(input_tensor)
            label_min = int(class_label_region[obj_class, 0])
            label_max = int(class_label_region[obj_class, 1] + 1)
            pre_image = pre_image[:, label_min:label_max, :, :] \
                .argmax(1) + label_min
            pre_sample = np.zeros_like(pre_test[0])
            pre_sample = pre_image[:, pos[:, 0], pos[:, 1]]

            pre_test[idx_sample] = pre_sample[:, None][0]
            pre_set[idx_sample] = pre_sample.T
            if idx_sample % 100 == 0:
                print('finish point segments: ', idx_sample, '/', len(s_test))
    return pre_test


def main(config):
    """
    Function for testing
    :param config: configuration for training -
        need only following keys for testing:
        'pt_model_path': pre-trained model path to be used for predictions
    """
    # reading class names of the ShapeNet dataset
    class_name = np.genfromtxt('all_object_categories.txt', dtype='U')[:, 0]

    # reading the test dataset from disk
    dataset_path = "shapenet_prepared.h5"
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

    # reading the trained model
    ckpt = Path(config['pt_model_path'])
    model = MultiScaleUNet()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    # creating test split
    test_dataset = ShapeNetPartDataset(split='test')
    print(test_dataset.__len__())

    # creating new file to store results
    result_path = 'ShapeNet_testing_result.hdf5'
    pre_test = test(s_test, p_test, x_test, y_test, model,
                    class_label_region, test_dataset, result_path)

    # calculate iou for each shape
    iou_shape = np.zeros(len(s_test))
    for idx_sample, pre_sample, gt_sample, obj_class in \
            zip(range(len(s_test)), pre_test, s_test, y_test):

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

        if idx_sample % 100 == 0:
            print('finish iou calculation: ', idx_sample, '/', len(s_test))

    print('iou_instance =', iou_shape.mean())

    iou_class = np.zeros(16)
    for obj_class in range(16):
        iou_obj_class = iou_shape[y_test[:] == obj_class]
        iou_class[obj_class] = iou_obj_class.mean()
    print('iou_class =', iou_class.mean())

    for obj_class in range(16):
        print('class', obj_class, ', class name:', class_name[obj_class],
              ",iou=", iou_class[obj_class])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='U-NET training configuration file path')
    parser.add_argument("--config_path", default="./config.yaml", type=str)
    args = parser.parse_args()
    # import the configuration file
    config = {}
    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # train
    main(config)
