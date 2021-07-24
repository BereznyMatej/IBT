"""
Thesis: 
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny 
File:
    train.py
Description:
    Script initiating the training procress. 
"""

import torch
import os
import json
import sys
import argparse
from torch import nn
from unet.logger import Logger
from torch.utils.data import DataLoader
from dataset.dataloader import get_dataset
from unet.trainer import UNetTrainer, UNetTrainerV2
from unet.loss import DC_and_BCE_loss
from unet.metrics import DiceCoefficient

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--load_weights', '-l', action='store_true')
parser.add_argument('--skip_validation', action='store_true')
parser.add_argument('--sparse_ann', action='store_true')

args = parser.parse_args()

net_name = args.model
data_folder = args.data

data_args = {'compute_masks': args.sparse_ann}

# Getting the dataset object
train, valid = get_dataset(f"../preprocessed/{data_folder}", data_args)

# Initializing DataLoader with set batch size and workers
train_loader = DataLoader(train, batch_size=10, num_workers=4, shuffle=True)
if not args.skip_validation:
    valid_loader = DataLoader(valid, batch_size=10, num_workers=4)
else:
    valid_loader = None
# Setting the trainer mode
mode = 'tune' if args.skip_validation else 'train'

val_iter = len(valid)
train_iter = len(train)

json_path = os.path.join('../pretrained_weights', net_name+'.json')

# Creating .json file storing basic information about current status of model
if os.path.exists(json_path):
    with open(json_path) as fp:
        json_dict = json.load(fp)
else:
    json_dict = {'max_epoch': 200,
                 'epoch': 0,
                 'index': 0,
                 'train_iter': train_iter,
                 'val_iter': val_iter,
                 'best_loss': 100000,
                 'epochs_no_improvement': 0,
                 'patience': 100}
    with open(json_path, 'w') as fp:
        json.dump(json_dict, fp, sort_keys=True, indent=4)

if train_iter != json_dict['train_iter'] or val_iter != json_dict['val_iter']:
    json_dict['train_iter'] = train_iter
    json_dict['val_iter'] = val_iter

# Creating Logger object
logger = Logger(folder='../pretrained_weights',
                net_name=net_name,
                json_dict=json_dict)
logger.log_sample_size(train, valid)

# Combined loss function with batch dice
loss = DC_and_BCE_loss({}, {'batch_dice': True, 'do_bg': True, 'smooth': 0})

# Checking if there is need to pass masks into the loss function
trainer_class = UNetTrainer if args.sparse_ann else UNetTrainerV2

# Creating the Trainer object with set parameters
trainer = trainer_class(start_epoch=json_dict['epoch'],
                        end_epoch=json_dict['max_epoch'],
                        criterion=loss,
                        metric=DiceCoefficient(),
                        logger=logger,
                        model_name=args.model,
                        momentum=0.95,
                        load=args.load_weights,
                        learning_rate=0.001,
                        mode=mode)
# Starting the training process
trainer.fit(train_loader, valid_loader)
