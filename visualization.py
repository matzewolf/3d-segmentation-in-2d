import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def visualize(idx_class: int, idx_class_sample: int,
              live: bool = True, export: bool = False):
    """
    Function for visualization of the model prediction and the ground truth
    !! test.py should be executed to store the test results before this file !!

    :param idx_class: class indexes from 0 to 15
    :param idx_class_sample: particular sample of chosen class
    :param live: If True, the visualizations will be shown live in a browser or
        within a Jupyter notebook.
    :param export: If True, the visualizations will be exported as separate
        HTML files.
    """
    result_path = Path('ShapeNet_testing_result.hdf5')
    class_name = np.genfromtxt('all_object_categories.txt', dtype='U')[:, 0]
    current_palette = sns.color_palette('bright', 10)

    with h5py.File(result_path, 'r') as f:
        x_test = f['x_test']
        y_test = f['y_test'][:]
        s_test = f['s_test']
        pre_test = f['pre_test']

        idx_sample_list = np.where(y_test == idx_class)[0]
        idx_sample = idx_sample_list[idx_class_sample]
        label_min = s_test[idx_sample_list].min()
        label_max = s_test[idx_sample_list].max()
        print(f'Class: {class_name[idx_class]}, sample id: {idx_sample}')

        x_pt = x_test[idx_sample]
        s_pt = s_test[idx_sample] - label_min
        pre_pt = pre_test[idx_sample] - label_min

    fig = go.Figure()
    for i_seg in range(label_max - label_min + 1):
        idxs = np.where(s_pt == i_seg)[0]
        color = current_palette.as_hex()[i_seg]
        fig.add_trace(go.Scatter3d(x=x_pt[idxs, 0],
                                   y=x_pt[idxs, 1],
                                   z=x_pt[idxs, 2],
                                   mode='markers',
                                   marker=dict(size=5, color=color)))
    if live:
        fig.show()
    if export:
        fig.write_html(
            f"ground_truth_{class_name[idx_class]}_{idx_class_sample}.html")

    fig = go.Figure()
    for i_seg in range(label_max - label_min + 1):
        idxs = np.where(pre_pt == i_seg)[0]
        color = current_palette.as_hex()[i_seg]
        fig.add_trace(go.Scatter3d(x=x_pt[idxs, 0],
                                   y=x_pt[idxs, 1],
                                   z=x_pt[idxs, 2],
                                   mode='markers',
                                   marker=dict(size=5, color=color)))
    if live:
        fig.show()
    if export:
        fig.write_html(
            f"prediction_{class_name[idx_class]}_{idx_class_sample}.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualization')
    parser.add_argument('idx_class', type=int)
    parser.add_argument('idx_class_sample', type=int)
    parser.add_argument('--no-plot', action='store_false')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    visualize(args.idx_class, args.idx_class_sample,
              live=args.no_plot, export=args.export)
