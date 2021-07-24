"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    preprocessor.py
Description:
    Class implementation for data preprocessing.
"""
import os
import sys
import numpy as np
import json
from scipy.interpolate import interp2d
from scipy.signal import argrelmax
from dataset.utils import *
import skimage.transform as skTrans


class Preprocessor:
    """Basic preprocessor class. Implements operations such as cropping, normalization,
    and padding.
    """
    def __init__(self, root_dir, patch_size, crop,
                 pad_type, look_for_labels, spacing,
                 normalize=True):
        """Preprocessor constructor.
        
        Args:
            root_dir (string): path to folder containing raw data
            patch_size (tuple of ints): patch size for padding and croping calculation
            crop (int): crop level: 0 - no crop, 1 - crop around air, 2 - crop around bone
            pad_type (string): padding type: 'zero_pad' or 'extrapolate'
            look_for_labels (bool): determines if labels will be ignored or not
            spacing (tuple of floats): spacing used in resampling
            normalize (bool, optional): determines if z-score normalization would be applied. Defaults to True.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.pad_type = pad_type
        self.crop_teeth = crop
        self.look_for_labels = look_for_labels
        self.spacing = spacing
        self.do_normalization = normalize
        self.transforms = []

    def _gen_threshold(self, volume, initial_threshold):
        """Computes histogram-based threshold from soft tissue peak and initial threshold.

        Args:
            volume (np.array): 3d array representing data volume
            initial_threshold ([type]): average difference between soft and bone tissue, 'd' coefficient in thesis

        Returns:
            float: computed threshold
        """
        data = volume.flatten()
        n, bins = np.histogram(data, 256)
        x = np.linspace(np.min(data), np.max(data), num=256)

        ind_max = argrelmax(n, order=1)
        x_max = x[ind_max]
        y_max = n[ind_max]
        index_first_max = np.argmax(y_max)

        return x_max[index_first_max] + initial_threshold

    def __normalize(self, volume, label=None):
        """Clips the volume to range (1%,99%) and applies z-score normalization.

        Args:
            volume (np.array): 3d array representing data volume
            label (np.array, optional): 3d array representing label volume. Defaults to None.

        Returns:
            np.array: normalized data and label volumes
        """
        upper_bound = np.percentile(volume, 99)
        lower_bound = np.percentile(volume, 1)
        mask = (volume > lower_bound) & (volume < upper_bound)
        volume = np.clip(volume, lower_bound, upper_bound)
        mn = volume[mask].mean()
        sd = volume[mask].std()
        volume = (volume - mn) / sd

        if label is not None:
            label = np.where(label > 0.1, 1, 0).astype(np.uint8)

        return volume, label

    def _crop(self, volume, label=None,
              initial_threshold=0.15, crop_teeth=2):
        """Crops the unimportant areas from CT volumes.
        
        Args:
            volume (np.array): 3d array representing data volume.
            label (np.array, optional): 3d array representing label volume. Defaults to None.
            initial_threshold (float, optional): average difference between soft and bone tissue. Defaults to 0.15.
            crop_teeth (bool, optional): crop level: 0 - no crop, 1 - crop around air, 2 - crop around bone. Defaults to 2.

        Returns:
            np.array: cropped data and label volumes.
        """
        # No cropping necessary.
        if crop_teeth == 0:
            return volume, label
        
        # Sanity check.
        if label is not None:
            if label.size != volume.size:
                sys.exit(f"Unequal sizes between annotations and volume! {label.shape} != {volume.shape}")

        threshold = self._gen_threshold(volume, initial_threshold)
        # Creating mask from bone regions. 
        if crop_teeth == 2:
            mean = np.nanmean(np.where((volume > threshold),
                              volume, np.nan), axis=(0, 1))
            image = volume[:, :, np.nanargmax(mean)]
            mask = np.where((image > threshold), image, 0) == 0
        # Creating mask from non-air regions.
        else:
            mean = np.nanmean(volume, axis=(0, 1))
            image = volume[:, :, np.argmax(mean)]
            mask = image == 0

        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)

        # Calculating new smallest possible dimensions that also match the patch size
        new_dims = get_new_dimensions((bottom_right[0] - top_left[0],
                                       bottom_right[1] - top_left[1],
                                       volume.shape[2]),
                                      dims=list(self.patch_size))
        width, length, depth = new_dims

        # Checking if new computed dimensions arent larger than original volume
        if width > volume.shape[0]:
            width = volume.shape[0]
        if length > volume.shape[1]:
            length = volume.shape[1]
        if depth > volume.shape[2]:
            depth = volume.shape[2]

        # Additional smoothing of cropped area boundaries
        x_range = (width - (bottom_right[0] - top_left[0]))
        y_range = (length - (bottom_right[1] - top_left[1]))

        uneven_width = x_range % 2
        uneven_length = y_range % 2

        top_left[0] -= x_range // 2
        top_left[1] -= y_range // 2

        if top_left[0] < 0:
            extra = abs(top_left[0])
            top_left[0] = 0
            if bottom_right[0] + extra <= volume.shape[0]:
                bottom_right[0] += extra
            else:
                bottom_right[0] = volume.shape[0]

        if top_left[1] < 0:
            extra = abs(top_left[1])
            top_left[1] = 0
            if bottom_right[1] + extra <= volume.shape[1]:
                bottom_right[1] += extra
            else:
                bottom_right[1] = volume.shape[1]

        bottom_right[0] += x_range // 2
        bottom_right[1] += y_range // 2

        if bottom_right[0] > volume.shape[0]:
            extra = bottom_right[0] - volume.shape[0]
            bottom_right[0] = volume.shape[0]
            if top_left[0] - extra >= 0:
                top_left[0] -= extra
            else:
                top_left[0] = 0

        if bottom_right[1] > volume.shape[1]:
            extra = bottom_right[1] - volume.shape[1]
            bottom_right[1] = volume.shape[1]
            if top_left[1] - extra >= 0:
                top_left[1] -= extra
            else:
                top_left[0] = 0

        if uneven_width:
            if top_left[0] - uneven_width >= 0:
                top_left[0] -= uneven_width
            elif bottom_right[0] + uneven_width <= volume.shape[0]:
                bottom_right[0] += uneven_width
        if uneven_length:
            if top_left[1] - uneven_length >= 0:
                top_left[1] -= uneven_length
            elif bottom_right[1] + uneven_length <= volume.shape[1]:
                bottom_right[1] += uneven_length

        # Cropping the volume.
        croped_volume = volume[top_left[0]:bottom_right[0],
                               top_left[1]:bottom_right[1], :]
        print(f"Cropping to: {croped_volume.shape},", end=' ')
        # Cropping the label. 
        if label is not None:
            croped_label = label[top_left[0]:bottom_right[0],
                                 top_left[1]:bottom_right[1], :]
        else:
            return croped_volume, None
        return croped_volume, croped_label

    def _pad(self, volume, new_dims, label=None):
        """Pads the area around CT volume to match the shape defined in new_dims.

        Args:
            volume (np.array): 3d array representing data volume.
            new_dims (tuple of ints): new shape to be padded to.
            label (np.array, optional): 3d array representing label volume. Defaults to None.

        Returns:
            np.array: padded data and label volumes
        """
        width, height, depth = volume.shape

        x_pad_left = (new_dims[0] - width) // 2
        x_pad = (x_pad_left, x_pad_left + (new_dims[0] - width) % 2)
        y_pad_left = (new_dims[1] - height) // 2
        y_pad = (y_pad_left, y_pad_left + (new_dims[1] - height) % 2)
        z_pad_left = (new_dims[2] - depth) // 2
        z_pad = (z_pad_left, z_pad_left + (new_dims[2] - depth) % 2)

        padded_volume = np.zeros(new_dims)

        if self.pad_type == 'zero_pad':
            padded_volume = np.pad(volume, (x_pad, y_pad, z_pad),
                                  mode='constant', constant_values=0)
        else:
            padded_volume = skTrans.resize(volume, new_dims,
                                          order=1, preserve_range=True)
        print(f"Padded to: {padded_volume.shape},", end=' ')
        if label is not None:
            if self.pad_type == 'zero_pad':
                padded_label = np.pad(label, (x_pad, y_pad, z_pad),
                                      mode='constant', constant_values=0)
            else:
                padded_label = skTrans.resize(label.astype(np.bool_),
                                              new_dims, order=0, preserve_range=True)
        else:
            return padded_volume, None
        return padded_volume, padded_label

    def save_train_test_split(self, split_args):
        """Splits the processed data into train/validation subsets and stores that 
        information along with augumentations and patch size. 

        Args:
            split_args (dict): arguments for gen_train_test_split function.
        """
        t_list, v_list = gen_train_test_split(**split_args)
        split_dict = {'train': t_list,
                      'valid': v_list,
                      'type': self.type,
                      'transforms': self.transforms,
                      'patch_size': self.patch_size}
        with open(f"{split_args['root_dir']}/settings.json", 'w') as fp:
            json.dump(split_dict, fp)

    def __extract(self, item_path):
        """Extracts the 3D numpy array from CT scans in various formats.

        Args:
            item_path (string): path to the CT scan.

        Returns:
            np.array: extracted 3D array.
        """
        if os.path.isdir(item_path):
            item = load_from_folder(item_path, self.spacing)
        elif '.nii.gz' in item_path:
            item = load_nifty(item_path, self.spacing)
        elif '.raw' in item_path:
            item = load_raw(item_path, self.spacing)
        else:
            return None
        return item

    def __call__(self, item):
        """Applies cropping, normalization and padding operations on
        given item.

        Args:
            item (string): path to the processed item.

        Returns:
            np.array: processed data and label volumes.
        """
        print(f"Loading file {item}...")
        try:
            start = item.rindex('_')
            end = item.index('.')
        except:
            return None, None

        if self.look_for_labels:
            labels_folder_path = os.path.join(self.root_dir, 'labels')
            label_list = os.listdir(labels_folder_path)
            label_name = [i for i in label_list if item[start:end] in i]
            label_path = os.path.join(labels_folder_path, label_name[0])

        if self.look_for_labels and not label_name:
            print(f"Missing label for file {item}")
            return None, None

        item_path = os.path.join(self.root_dir, item)

        volume = self.__extract(item_path)

        if volume is None:
            print(f"Data format of {item} is not supported...",
                  end=' ')
            return None, None

        if self.look_for_labels:
            label = self.__extract(label_path)
            if label is None:
                print(f"Data format of {label_name} is not supported...",
                      end=' ')
                return None, None
        else:
            label = None
        print(f"Original Shape: {volume.shape},", end=" ")
        volume, label = self._crop(volume, label=label,
                                   crop_teeth=self.crop_teeth)
        new_dims = get_new_dimensions(volume.shape, list(self.patch_size))
        volume, label = self._pad(image=volume,
                                  label=label,
                                  new_dims=new_dims)
        if self.do_normalization:
            volume, label = self.__normalize(volume, label)
            
        return volume, label


class PreprocessorSupervised(Preprocessor):
    """PreprocessorSupervised class. Extends the original class by additionaly 
    processing labels and saving the processed data.
    """
    def __init__(self, root_dir='../raw', patch_size=(64, 64, 32), crop=2,
                 pad_type='zero_pad', look_for_labels=True,
                 spacing=(0.8, 0.8, 0.8), **kwargs):
        """PreprocessorSupervised constructor, extending the Preprocessor constructor by 
        defining augumentations and level of supervision.

        Args:
            root_dir (string): path to folder containing raw data. Defaults to '../raw'.
            patch_size (tuple of ints): patch size for padding and croping calculation. Defaults to (64, 64, 32).
            crop (int): crop level: 0 - no crop, 1 - crop around air, 2 - crop around bone. Defaults to 2.
            pad_type (string): padding type: 'zero_pad' or 'extrapolate'. Defaults to 'zero_pad'.
            look_for_labels (bool): determines if labels will be ignored or not. Defaults to True.
            spacing (tuple of floats): spacing used in resampling. Defaults to (0.8, 0.8, 0.8).
        """
        super(PreprocessorSupervised, self).__init__(root_dir=root_dir,
                                                     patch_size=patch_size,
                                                     pad_type=pad_type,
                                                     look_for_labels=look_for_labels,
                                                     crop=crop,
                                                     spacing=spacing)
        self.transforms = [{'name': 'Mirroring',
                            'execution_probability': 0.5},
                           {'name': 'Rotate',
                            'execution_probability': 0.5},
                           {'name': 'ElasticDeformation',
                            'execution_probability': 0.5},
                           {'name': 'Gamma',
                            'execution_probability': 0.5},
                           {'name': 'Contrast',
                            'execution_probability': 0.3}]
        self.type = "supervised"

    def __call__(self, dest_dir, item, index):
        """Extends the original function by additionaly saving the processed scan.

        Args:
            dest_dir (string): destination, where to save processed item.
            item (string): path to the processed item.
            index (int): index of processed item.
        """
        image, label = super(PreprocessorSupervised, self).__call__(item)

        if image is None:
            print("Skipping item.")
            return

        print(f"Saving file...", end='')
        np.savez(os.path.join(dest_dir, "volume_{:03}.npz".format(index)),
                 image, allow_pickle=True)
        if not os.path.isdir(os.path.join(dest_dir, 'labels')):
            os.mkdir(os.path.join(dest_dir, 'labels'))
        np.savez(os.path.join(dest_dir, 'labels',
                              "label_{:03}.npz".format(index)),
                 label.astype(np.uint8), allow_pickle=True)
        print("Done.\n")

        param_dict = {'root_dir': dest_dir,
                      'valid_ratio': 0.3}


class PreprocessorSelfSupervised(Preprocessor):
    """PreprocessorSelfSupervised class. Extends the original class by not 
    processing the labels and saving the processed data.
    """
    def __init__(self, root_dir='../raw', patch_size=(64, 64, 32), crop=2,
                 pad_type='zero_pad', look_for_labels=False,
                 spacing=(0.8, 0.8, 0.8), **kwargs):
        """PreprocessorSelfSupervised constructor, extending the Preprocessor constructor by 
        defining augumentations and level of supervision.

        Args:
            root_dir (string): path to folder containing raw data. Defaults to '../raw'.
            patch_size (tuple of ints): patch size for padding and croping calculation. Defaults to (64, 64, 32).
            crop (int): crop level: 0 - no crop, 1 - crop around air, 2 - crop around bone. Defaults to 2.
            pad_type (string): padding type: 'zero_pad' or 'extrapolate'. Defaults to 'zero_pad'.
            look_for_labels (bool): determines if labels will be ignored or not. Defaults to True.
            spacing (tuple of floats): spacing used in resampling. Defaults to (0.8, 0.8, 0.8).
        """
        super(PreprocessorSelfSupervised, self).__init__(root_dir=root_dir,
                                                         patch_size=patch_size,
                                                         pad_type=pad_type,
                                                         look_for_labels=look_for_labels,
                                                         crop=crop,
                                                         spacing=spacing)
        self.transforms = [{'name': 'Mirroring', 'execution_probability': 0.5}]
        self.type = "self-supervised"

    def __call__(self, dest_dir, item, index):
        """Extends the original function by additionaly saving the processed scan.

        Args:
            dest_dir (string): destination, where to save processed item.
            item (string): path to the processed item.
            index (int): index of processed item.
        """
        image, _ = super(PreprocessorSelfSupervised, self).__call__(item)
        if image is None:
            print("Skipping item.")
            return

        print(f"Saving file...", end='')
        np.savez(os.path.join(dest_dir, "volume_{:03}.npz".format(index)),
                 image, allow_pickle=True)
        print("Done.")


def get_preprocessor(preprocessor_args, no_label=False):
    """Simple method for selecting the right type of preprocessor class for target data

    Args:
        preprocessor_args (dict): [description]
        no_label (bool, optional): disables label preprocessing. Defaults to False.

    Returns:
        Preprocessor: returns initialzed preprocessor
    """
    if no_label:
        preprocessor = PreprocessorSelfSupervised
    else:
        preprocessor = PreprocessorSupervised
    return preprocessor(**preprocessor_args)
