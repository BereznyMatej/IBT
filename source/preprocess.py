"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    preprocess.py
Description:
    Control script handling the preprocessing raw medical data.
"""
import argparse
import os
from dataset.preprocessor import get_preprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--crop', '-c', type=int)
parser.add_argument('--input', '-i', type=str)
parser.add_argument('--patch_size', '-p', type=int, default=64)
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--extrapolate', '-e', action='store_true')
parser.add_argument('--no_label', '-n', action='store_true')
parser.add_argument('--spacing', '-s', type=float, default=-1.0)

args = parser.parse_args()

# Setting the type of the padding
pad_type = 'extrapolate' if args.extrapolate else 'zero_pad'
output_dir = f"../preprocessed/{args.model}"

# Arguments for Preprocessor
preprocessor_args = {'root_dir': args.input,
                     'patch_size': (args.patch_size,
                                    args.patch_size,
                                    args.patch_size),
                     'pad_type': pad_type,
                     'crop': args.crop,
                     'spacing': (args.spacing, args.spacing, args.spacing)}

# Arguments for train/test split generator
split_args = {'root_dir': output_dir,
              'valid_ratio': 0.3}

# Getting the right preprocessor class
preprocessor = get_preprocessor(preprocessor_args, no_label=args.no_label)

if not os.path.isdir('../preprocessed'):
    os.mkdir('../preprocessed')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

index = 0
# Iterate over every file in folder with raw data
for item in os.listdir(args.input):
    # Skip labels directory
    if 'labels' in item:
        continue
    preprocessor(output_dir, item, index)
    index += 1
# Create the .json file containing info about preprocessed data
preprocessor.save_train_test_split(split_args)
