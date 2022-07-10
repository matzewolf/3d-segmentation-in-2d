# Learning to segment 3D point clouds in 2D image space

[![pre-commit.ci status](
    https://results.pre-commit.ci/badge/github/matzewolf/3d-segmentation-in-2d/main.svg)
](https://results.pre-commit.ci/latest/github/matzewolf/3d-segmentation-in-2d/main)

This repository is a community implementation of the paper
[Learning to Segment 3D Point Clouds in 2D Image Space](https://arxiv.org/abs/2003.05593)
by Lyu et al. using the PyTorch framework. The official TensorFlow implementation by the authors can be found
[here](https://github.com/Zhang-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space).

## Getting started

### Install environment

Install the conda environment with all necessary packages by
```commandline
conda env create -f environment.yml
```
and activate it by
```commandline
conda activate 3d-seg-in-2d
```

### Download ShapeNet Part dataset

```commandline
./download_data.sh
```
will download and extract the ShapeNet Part dataset from Stanford University into the directory `hdf5_data`.

## Contributing

To install the pre-commit hooks, run `pre-commit install`.
