# Learning to Segment 3D Point Clouds in 2D Image Space

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

### Project 3D point clouds into 2D image space

```commandline
python prepare_dataset.py
```

## Training

Create a custom configuration file `custom_config.yaml` or use the default `config.yaml`
to determine the hyperparameters and other settings. Run
```commandline
python train_unet.py --config_path custom_config.yaml
```
or simply `python train_unet.py` to use the default configuration.

### Configuration options

- `device`: Enable CUDA acceleration by setting the device in the configuration to `device: 'cuda:0'` or similar.
  MPS for Apple Silicon is also supported, set the device to `device: 'mps'`.

## Evaluation

```commandline
python test.py --model_path <MODEL_PATH>
```
to evaluate the model located at `<MODEL_PATH>`.

## Contributing

You can run the pre-commit hooks locally with `pre-commit run`.
