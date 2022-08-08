import argparse

import h5py
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import yaml

current_palette = sns.color_palette('bright', 10)


def visualize(config):
    """
    Function for visualization of the model prediction and the ground truth
    !! test.py should be executed to store the test results before this file !!

    :param config: keys related with visualization:
        'idx_class': class indexes from 0 to 15
        'idx_class_sample': particular sample of chosen class
        'export': True if you want to export the visualizations
            as separate .html files
        'on_Jupyter': True if you're working on a Jupyter Notebook,
            so the visualizations are visible withing the cell that
            is executed. If it's kept as True while not working
            with a Jupyter Notebook, default browser should open
            the visualizations automatically
    """

    ###########################################################
    # ['Airplane' 'Bag' 'Cap' 'Car' 'Chair' 'Earphone' 'Guitar'
    #   'Knife' 'Lamp' 'Laptop' 'Motorbike' 'Mug' 'Pistol'
    #  'Rocket' 'Skateboard' 'Table'] #
    ###########################################################
    result_path = 'ShapeNet_testing_result.hdf5'
    class_name = np.genfromtxt('all_object_categories.txt', dtype='U')[:, 0]

    print("loading testing data")
    f = h5py.File(result_path, 'r')
    x_test = f['x_test']
    y_test = f['y_test'][:]
    s_test = f['s_test']
    pre_test = f['pre_test']

    idx_class = config['idx_class']
    idx_class_sample = config['idx_class_sample']

    idx_sample_list = np.where(y_test == idx_class)[0]
    idx_sample = idx_sample_list[idx_class_sample]
    label_min = s_test[idx_sample_list].min()
    label_max = s_test[idx_sample_list].max()
    print('Class_name:', class_name[idx_class],
          ', test sample id:', idx_sample)

    x_pt = x_test[idx_sample]
    s_pt = s_test[idx_sample] - label_min
    pre_pt = pre_test[idx_sample] - label_min

    fig = go.Figure()
    for i_seg in range(label_max - label_min + 1):
        idxs = np.where(s_pt == i_seg)[0]
        color = current_palette.as_hex()[i_seg]
        fig.add_trace(go.Scatter3d(x=x_pt[idxs, 0],
                      y=x_pt[idxs, 1], z=x_pt[idxs, 2],
                      mode='markers',
                      marker=dict(size=5, color=color)))
    if config["on_Jupyter"]:
        fig.show()
    if config["export"]:
        fig.write_html("GroundTruth.html")

    fig = go.Figure()
    for i_seg in range(label_max - label_min + 1):
        idxs = np.where(pre_pt == i_seg)[0]
        color = current_palette.as_hex()[i_seg]
        fig.add_trace(go.Scatter3d(x=x_pt[idxs, 0],
                      y=x_pt[idxs, 1], z=x_pt[idxs, 2],
                      mode='markers',
                      marker=dict(size=5, color=color)))
    if config["on_Jupyter"]:
        fig.show()
    if config["export"]:
        fig.write_html("Prediction.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='visualization')
    parser.add_argument("--config_path", default="./config.yaml", type=str)
    args = parser.parse_args()
    # import the configuration file
    config = {}
    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    visualize(config)
