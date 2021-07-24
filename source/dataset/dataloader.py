"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    dataloader.py
Description:
    Class implementation for dataloading and applying transformations.
"""
import torch
import os
import numpy as np
import random
import copy
import json
from torch.utils.data import Dataset
from dataset.utils import get_patches
from unet.utils import *
from dataset.transforms import Transformer


class Teeth(Dataset):
    """Generic dataset class.
    """
    def __init__(self, patch_size, transform, mode):
        """Teeth dataset constructor.

        Args:
            patch_size (tuple of ints): 
            transform (torchvision.transforms.Compose): Compose object with all transformations
            mode (string): 'training' or 'validation'
        """
        self.transform = transform
        self.mode = mode
        self.patch_size = patch_size

    def __getitem__(self, index):
        """Selects one item from dataset and applies transforms (if present).

        Args:
            index (int): index of expected item

        Returns:
            np.array: augmented training sample with dims (C,X,Y,Z) 
        """
        if self.transform:
            sample = copy.deepcopy(self.data[index])
            sample = np.expand_dims(self.transform(sample), axis=1)
        else:
            sample = np.expand_dims(copy.deepcopy(self.data[index]), axis=1)
        return sample

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return self.data.shape[0]


class TeethSupervised(Teeth):
    """Extends the original class by adding the option to compute masks for
    sparse annotations.
    """
    def __init__(self, root_dir, transform=None, filename_list=None,
                 mode="training", compute_masks=False,
                 patch_size=(32, 32, 32), **kwargs):
        """Extends the original incomplete implementation.

        Args:
            root_dir (string): folder with preprocessed files.
            transform (torchvision.transforms.Compose): Compose object with all transformations
            filename_list (list, optional): list with filenames of specific subset. Defaults to None.
            mode (str, optional): 'training' or 'validation' . Defaults to "training".
            compute_masks (bool, optional):  Defaults to False.
            patch_size (tuple, optional): Defaults to (32, 32, 32).
        """
        super(TeethSupervised, self).__init__(transform=transform,
                                              mode=mode,
                                              patch_size=patch_size)
        self.compute_masks = compute_masks
        self.data = self.__load(root_dir, filename_list)

    def __load(self, root_dir, filename_list):
        """Loads the preprocessed data into the memory and slices them into patches.

        Args:
            root_dir (string): folder with preprocessed files.
            filename_list (list, optional): list with filenames of specific subset. Defaults to None.

        Returns:
            np.array: array with patches generated from preprocessed data. Patch dims:
            (C,X,Y,Z), where C = 2/3 -> data, label, (mask, optional)
        """
        patches = get_patches(root_dir, filename_list=filename_list,
                              compute_masks=self.compute_masks,
                              shape=self.patch_size)
        return np.array(patches)

    def __getitem__(self, index):
        """Extends the original implementation by adding mask required for sparse annotations.

        Args:
            index (int): index of expected item

        Returns:
            dict: dictionary including data patch, label patch and appropriate mask.
        """
        sample = super(TeethSupervised, self).__getitem__(index)

        if self.compute_masks:
            x, y, mask = sample
        else:
            x, y = sample
            # Mask is set to array(1), since DataLoader does not accept None
            mask = np.array([1])

        return {'x': x, 'y': y, 'mask': mask}


class TeethSelfSupervised(Teeth):
    """Extends the original class by adding custom image distorting operations
    for image restoration task.
    """
    def __init__(self, root_dir, transform=None, filename_list=None,
                 local_rate=0.5, nonlinear_rate=0.9,
                 outpaint_rate=0.8, mode="training",
                 patch_size=(32, 32, 32), **kwargs):
        """Extends the original incomplete implementation.

        Args:
            root_dir (string): folder with preprocessed files.
            transform (torchvision.transforms.Compose): Compose object with all transformations
            filename_list (list, optional): list with filenames of specific subset. Defaults to None.
            local_rate (float, optional): chance of applying local pixel shuffling. Defaults to 0.5.
            nonlinear_rate (float, optional): chance of applying nonlinear transformation. Defaults to 0.9.
            outpaint_rate (float, optional): chance of applying outpainting. Defaults to 0.8.
            mode (str, optional): 'training' or 'validation'. Defaults to "training".
            patch_size (tuple, optional): Defaults to (32, 32, 32).
        """
        super(TeethSelfSupervised, self).__init__(mode=mode,
                                                  transform=transform,
                                                  patch_size=patch_size)
        self.local_rate = local_rate
        self.nonlinear_rate = nonlinear_rate
        self.outpaint_rate = outpaint_rate
        self.inpaint_rate = 1.0 - outpaint_rate
        self.data = self.__load(root_dir, filename_list)

    def __load(self, root_dir, filename_list):
        """Loads the preprocessed data into the memory and slices them into patches.

        Args:
            root_dir (string): folder with preprocessed files.
            filename_list (list, optional): list with filenames of specific subset. Defaults to None.

        Returns:
            np.array: array with patches generated from preprocessed ata.
        """
        patches = get_patches(root_dir, filename_list=filename_list,
                              load_labels=False)
        labels = copy.deepcopy(patches)
        data = np.concatenate((np.array(patches), np.array(labels)), axis=1)
        return data

    def _gen_label(self, img):
        """Applies image distortion operations on single patch.

        Args:
            img (np.array): patch

        Returns:
            np.array: distorted patch with dims (C,X,Y,Z), where C = 2 -> distorted patch, original patch.
        """
        # local pixel shuffling
        img[0] = local_pixel_shuffling(img[0], prob=self.local_rate)
        # nonlinear transformation
        img[0] = nonlinear_transformation(img[0], self.nonlinear_rate)

        # either inpainting or outpainting
        if random.random() < self.outpaint_rate:
            if random.random() < self.inpaint_rate:
                img[0] = image_in_painting(img[0])
            else:
                img[0] = image_out_painting(img[0])
        return img

    def __getitem__(self, index):
        """Extends the original implementation by applying distortion methods.

        Args:
            index (int): index of expected item

        Returns:
            dict: dictionary including data patch, label patch and array(1) mask.
        """
        sample = super(TeethSelfSupervised, self).__getitem__(index)
        x, y = self._gen_label(sample)
        # Mask is set to array(1), since DataLoader does not accept None
        return {'x': x, 'y': y, 'mask': np.array([1])}


def get_dataset(dir, loader_args):
    """Prepares the appropriate types of datasets for DataLoader.

    Args:
        dir (string): folder with preprocessed data.
        loader_args (dict): 

    Returns:
        Dataset: training and validation dataset.
    """
    with open(os.path.join(dir, 'settings.json')) as fp:
        settings = json.load(fp)

    transformer = Transformer(settings['transforms'])
    transforms = transformer.create_transform()

    if settings['type'] == 'supervised':
        loader = TeethSupervised
    else:
        loader = TeethSelfSupervised

    train = loader(root_dir=dir, filename_list=settings['train'],
                   patch_size=settings['patch_size'],
                   transform=transforms, **loader_args)
    valid = loader(root_dir=dir, filename_list=settings['valid'],
                   patch_size=settings['patch_size'], **loader_args)

    return train, valid
