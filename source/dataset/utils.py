"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    utils.py
Description:
    Various utility functions used for manipulating with data.
"""
import nibabel as nib
import numpy as np
import random
import pydicom
import sys
import os
import glob
import torch
import torch.nn as nn
import SimpleITK as sitk
import tempfile
from collections import Counter
from skimage import io
from nibabel.processing import resample_to_output

def load_from_folder(dir, spacing):
    """Extracts the 3D array from folder with either .png or .dcm files.

    Args:
        dir (string): folder path
        spacing (tuple of floats): target spacing used for resampling.

    Returns:
        np.array: 3D array with data.
    """
    item = io.imread_collection(os.path.join(dir,'*.png'))
    if not item:
        item = load_dicom(dir, spacing)
        return scale(item)
    return scale(item.concatenate().swapaxes(0,2))

    
def load_nifty(dir, spacing):
    """Extracts and resamples the 3D array from .nii file.

    Args:
        dir (string): folder path
        spacing (tuple of floats): target spacing used for resampling.

    Returns:
        np.array: 3D array with data.
    """
    
    image = nib.load(dir)
    if spacing[0] > 0:
        image = resample_to_output(image, spacing)

    data = image.get_fdata()    
    return scale(data)
    
def load_raw(dir, spacing):
    """Extracts and resamples the 3D array from .raw file without header.
    
    Filename must have specific formatting: *_W_H_D_DTYPE_INDEX.raw
        W: Width of image.
        H: Height of image.
        D: Depth of image.
        DTYPE: Data type (uchar,uint_16...).
        INDEX: index used for finding appropriate label files.

    Args:
        dir ([type]): [description]
        spacing ([type]): [description]

    Returns:
        np.array: 3D array with data.
    """
    index = 0
    data_type = None
    idk = []
    name = dir[dir.rindex('/'):dir.rindex('_')]
    while index < len(name):
        try:
            index = name.find('_',index)
            next = name.find('_',index+1)
        except:
            return None
        if index == -1:
            break
        elif next == -1:
            if name[index+1:] == 'uchar':
                data_type = sitk.sitkUInt8
            else:
                 data_type = sitk.sitkUInt16
            break

        idk.append(int(name[index+1:next]))
        index+= len(name[index:next])



    image = read_raw(dir,(tuple(idk)),data_type, spacing)
    return scale(np.transpose(sitk.GetArrayFromImage(image),(2,1,0)))

def read_raw(binary_file_name, image_size, sitk_pixel_type, image_spacing=(0.2,0.2,0.2),
             image_origin=None, big_endian=False):
    """
    Read a raw binary scalar image.

    Implementation taken from:
        https://simpleitk.readthedocs.io/en/master/link_RawImageReading_docs.html        
    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to
        be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else
        little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """

    pixel_dict = {sitk.sitkUInt8: 'MET_UCHAR',
                  sitk.sitkInt8: 'MET_CHAR',
                  sitk.sitkUInt16: 'MET_USHORT',
                  sitk.sitkInt16: 'MET_SHORT',
                  sitk.sitkUInt32: 'MET_UINT',
                  sitk.sitkInt32: 'MET_INT',
                  sitk.sitkUInt64: 'MET_ULONG_LONG',
                  sitk.sitkInt64: 'MET_LONG_LONG',
                  sitk.sitkFloat32: 'MET_FLOAT',
                  sitk.sitkFloat64: 'MET_DOUBLE'}
    direction_cosine = ['1 0 0 1', '1 0 0 0 1 0 0 0 1',
                        '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1']
    dim = len(image_size)
    header = ['ObjectType = Image\n'.encode(),
              ('NDims = {0}\n'.format(dim)).encode(),
              ('DimSize = ' + ' '.join([str(v) for v in image_size]) + '\n')
              .encode(),
              ('ElementSpacing = ' + (' '.join([str(v) for v in image_spacing])
                                      if image_spacing else ' '.join(
                  ['1'] * dim)) + '\n').encode(),
              ('Offset = ' + (
                  ' '.join([str(v) for v in image_origin]) if image_origin
                  else ' '.join(['0'] * dim) + '\n')).encode(),
              ('TransformMatrix = ' + direction_cosine[dim - 2] + '\n')
              .encode(),
              ('ElementType = ' + pixel_dict[sitk_pixel_type] + '\n').encode(),
              'BinaryData = True\n'.encode(),
              ('BinaryDataByteOrderMSB = ' + str(big_endian) + '\n').encode(),
              # ElementDataFile must be the last entry in the header
              ('ElementDataFile = ' + os.path.abspath(
                  binary_file_name) + '\n').encode()]
    fp = tempfile.NamedTemporaryFile(suffix='.mhd', delete=False)

    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    print(header)
    img = sitk.ReadImage(fp.name)
    os.remove(fp.name)
    return img

def transform_to_hu(medical_image, image):
    """Transforms the .dcm image into Hounsfield units.

    Args:
        medical_image (image): original .dcm image.
        image (np.array): data extracted from image.

    Returns:
        np.array: transformed array.
    """
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def load_dicom(dir):
    """Loads all dicom files in folder and merges them into the 3D volume.

    Args:
        dir (string): path to folder with .dcm files

    Returns:
        np.array: 3D array with data.
    """
    files = []
    for fname in glob.glob(os.path.join(dir, "*.dcm"), recursive=False):
        files.append(pydicom.dcmread(fname))
    
    if not files:
        return None

    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    slices = sorted(slices, key=lambda s: s.SliceLocation)
    
    ps = slices[0].PixelSpacing


    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        img3d[:, :, i] = transform_to_hu(s,s.pixel_array)

    return img3d

def scale(image):
    """Min-max scaling.

    Args:
        image (np.array): image that will be scaled

    Returns:
        np.array: scaled image.
    """
    return (image - np.min(image))/np.ptp(image)


def roundTo(x,number):
    """Rounds to the nearest higher multiple of 'number'.

    Args:
        x (int): number to be rounded. 
        number (int): 

    Returns:
        int: rounded number.
    """
    return (x + (number-1)) & ~(number-1)


def get_new_dimensions(old_dims, dims):
    """Calculates new dimensions as nearest higher multiple of 'dims'.

    Args:
        old_dims (tuple of ints): original dimensions
        dims (tuple of ints): 

    Returns:
        tuple of ints: new dimensions
    """
    for i in range(len(old_dims)):
        dims[i] = roundTo(old_dims[i],dims[i])
    return tuple(dims)


def gen_train_test_split(root_dir, valid_ratio=0.3):
    """Splits the dataset into training/validation subsets.

    Args:
        root_dir (string): folder with dataset.
        valid_ratio (float, optional): Size of validation subset compared to training subset. Defaults to 0.3.

    Returns:
        list: 2 lists containing names of files that are either in validation or training subset.
    """
    data_list = os.listdir(root_dir)
    
    if 'labels' in data_list:
        data_list.remove('labels')
    num_of_validation_samples = int(len(data_list)*valid_ratio)

    validation = random.sample(data_list, num_of_validation_samples)
    train = list((Counter(data_list)-Counter(validation)).elements())
    
    return train, validation

def create_mask(target):
    """Creates mask for valid annotations in label volume.

    Args:
        target (np.array): volume with labels

    Returns:
        np.array: mask with 1 for all valid annotations.
    """
    mask = np.zeros(target.shape)
    for i in range(target.shape[-1]):
        mask[:,:,i] = np.ones((target.shape[:-1])) if target[:,:,i].max() > 0.5 else np.zeros((target.shape[:-1]))
    return mask


def split_to_patches(subset, shape, stride=None):
    """Splits 

    Args:
        subset ([type]): [description]
        shape ([type]): [description]
        stride ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    kc, kh, kw = shape
    if not stride:
        dc, dh, dw = kc, kh, kw
    else: 
        dc, dh, dw = stride
    
    samples = list()

    for i in range(subset.shape[0]):
        patches = torch.tensor(np.expand_dims(subset[i], axis=0)).unfold(1, kc, dc) \
                                                                 .unfold(2, kh, dh) \
                                                                 .unfold(3, kw, dw)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, 1, kc, kh, kw)
        samples.append(patches.numpy())
    
    return np.concatenate(samples, axis=1), unfold_shape


def get_patches(root_dir, filename_list, compute_masks=False,
                load_labels=True, shape=(32,32,32), stride=None):

        patches = []
        
        for data in os.listdir(root_dir):
            if '.npz' in data :
                if data in filename_list:
                    with np.load(os.path.join(root_dir, data)) as file:                
                        v = file['arr_0']
                        subset = np.zeros((1 + load_labels + compute_masks, *v.shape))
                        subset[0] = v
                    if load_labels:
                        with np.load(os.path.join(root_dir, 'labels', f"label{data[6:]}")) as file:
                            l = file['arr_0']
                            subset[1] = l
                            if compute_masks:
                                m = create_mask(np.array(l))
                                subset[2] = m
                    subset, _ = split_to_patches(subset, shape)
                    patches.extend(subset)
        return patches
