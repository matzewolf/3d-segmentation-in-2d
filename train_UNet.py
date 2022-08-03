import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import ShapeNetPartDataset
from model import MultiSacleUNet


def train(model, train_dataloader, val_dataloader, device, config):
    # Declare loss and move to device
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    eval_loss = nn.CrossEntropyLoss()
    eval_loss.to(device)
    # Declare optimizer with learning rate given in config
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'])
    best_loss_val = np.inf
    # Keep track of running average of train loss for printing, and step loss for plotting
    train_loss_running = 0.
    training_log_dict = {}
    val_log_dict = {}

    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # Set model to train
            model.train()
            # Move batch to device
            ShapeNetPartDataset.move_batch_to_device(batch, device)
            # set optimizer gradients to zero, perform forward pass
            optimizer.zero_grad()
            predicted_part_label = model(batch['3d_points'])
            loss = criterion(predicted_part_label, batch['part_label'])
            loss.backward()
            optimizer.step()
            # Logging
            step_loss = loss.item()
            train_loss_running += step_loss

            # add the step loss to the logging dict
            if epoch not in training_log_dict.keys():
                training_log_dict[epoch] = []
            training_log_dict[epoch].append(step_loss)
            # print the running average trainign loss
            iteration = epoch * len(train_dataloader) + batch_idx
            if iteration % config['print_every_n'] == config[
                    'print_every_n'] - 1:
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: ', end="")
                print(f'{train_loss_running / config["print_every_n"]:.6f}')
                train_loss_running = 0.

            # Validation evaluation and logging
            if (iteration % config['validate_every_n'] == config[
                    'validate_every_n'] - 1) or (iteration % len(val_dataloader) ==0):
                # Set model to eval
                model.eval()
                # Evaluation on entire validation set
                loss_val = 0.
                for batch_val in val_dataloader:
                    # Move batch to device
                    ShapeNetPartDataset.move_batch_to_device(batch_val, device)
                    #  validation forward loss
                    with torch.no_grad():
                        prediction = model(batch_val['3d_points'])
                    # calculate validation step loss
                    loss_val += eval_loss(prediction,
                                          batch_val['part_label']).item()
                # get the validation epoch loss
                loss_val /= len(val_dataloader)
                # if end of epoch, save validation loss for logging
                if (iteration % len(val_dataloader) ==0):
                    val_log_dict[epoch] = loss_val
                # check if this is best validation loss
                if loss_val < best_loss_val:
                    torch.save(
                        model.state_dict(),
                        f'./runs/{config["experiment_name"]}/model_best.ckpt')
                    best_loss_val = loss_val

                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: ', end="")
                print(f'{loss_val:.6f} | best_val_loss: {best_loss_val:.6f}')


def main(config):
    """
    Function for training multiscale U-Net on ShapeNetPart

    :param config: configuration for training - has the following keys:
        'experiment_name': name of the experiment, checkpoint will be saved to
            folder "/runs/<experiment_name>"
        'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
        'batch_size': batch size for training and validation dataloaders
        'resume_ckpt': None if training from scratch, otherwise path to
            checkpoint (saved weights)
        'learning_rate': learning rate for optimizer
        'max_epochs': total number of epochs after which training should stop
        'print_every_n': print train loss every n iterations
        'validate_every_n': print validation loss and validation accuracy every
            n iterations
        'is_overfit': Training and validation done on the same small subset of
            the data, where a loss close to 0 means a good overfit. Mainly used
            for debugging.
    """

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNetPartDataset(
        path='shapenet_prepared.h5',
        split='train' if not config['is_overfit'] else 'overfit'
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataset = ShapeNetPartDataset(
        path='shapenet_prepared.h5',
        split='val' if not config['is_overfit'] else 'overfit'
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Instantiate model
    model = MultiSacleUNet()

    # Load model if resuming from checkpoint
    if config['resume_ckpt']:
        model.load_state_dict(torch.load(config['resume_ckpt'],
                                         map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'./runs/{config["experiment_name"]}').mkdir(
        exist_ok=True, parents=True)

    # save the configurations used for this experiment
    with open(f'./runs/{config["experiment_name"]}/used_config.yml',
              'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)


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
