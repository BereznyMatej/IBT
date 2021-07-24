"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    transforms.py
Description:
    Class implementation for transformations.
"""
import numpy as np
import importlib
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from torchvision.transforms import Compose

# fixed state for reproducibility
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

class Mirroring:
    """Modified version of mirroring augmentation for images with shape (CxWxHxD)
    
    Original implementation taken from: 
            https://github.com/MIC-DKFZ/batchgenerators/blob/de0a67d0f9916bb1bfdbe2aa77579774d2f2342b/batchgenerators/augmentations/spatial_transformations.py#L118
    """
    def __init__(self, random_state, execution_probability=0.5, **kwargs):
        """Randomly flips the patch in 0 and 1 axis. 
        
        Args:
            random_state 
            execution_probability (float, optional): probability of applying mirroring. Defaults to 0.5.
        """
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = execution_probability

    def __call__(self, m):

        sample_data = m[0]
        sample_seg = m[1]
        if m.shape[0] == 3:
            sample_mask = m[2]
        else:
            sample_mask = None

        if 0 in self.axes and self.random_state.uniform() < self.axis_prob:
            sample_data[:, :] = sample_data[:, ::-1]
            sample_seg[:, :] = sample_seg[:, ::-1]
            if sample_mask is not None:
                sample_mask[:, :] = sample_mask[:, ::-1]
        if 1 in self.axes and self.random_state.uniform() < self.axis_prob:
            sample_data[:, :, :] = sample_data[:, :, ::-1]
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
            if sample_mask is not None:
                sample_mask[:, :, :] = sample_mask[:, ::-1]
        

        m[0] = sample_data
        m[1] = sample_seg
        if sample_mask is not None:
            m[2] = sample_mask
        return m

class Gamma:
    """Modified version of gamma correction transformation for images with shape (CxWxHxD).
    
    Implementation taken from: 
            https://github.com/MIC-DKFZ/batchgenerators/blob/de0a67d0f9916bb1bfdbe2aa77579774d2f2342b/batchgenerators/augmentations/color_augmentations.py#L106
    """
    def __init__(self, random_state, gamma_range=(0.5,2),
                 epsilon=1e-7, execution_probability=0.5, **kwargs):
        """ Augments by changing 'gamma' of the image.

        Args:
            random_state ([type]): [description]
            gamma_range (tuple, optional): range to sample gamma from. If one value is smaller than 1 and the other one is
                                           larger then half the samples will have gamma <1 and the other >1 
                                           (in the inverval that was specified). Defaults to (0.5,2).
            epsilon (float, optional): for smoothing. Defaults to 1e-7.
            execution_probability (float, optional): probability of applying mirroring. Defaults to 0.5.
        """
        self.gamma_range = gamma_range
        self.epsilon = epsilon
        self.exec_prob = execution_probability
        self.random_state = random_state

    def __call__(self, m):

        if self.random_state.uniform() < self.exec_prob and self.gamma_range[0] < 1:
            gamma = np.random.uniform(self.gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
        minm = m[0].min()
        rnge = m[0].max() - minm
        m[0] = np.power(((m[0] - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm

        return m


class Rotate:
    """90 degree rotation augmentation of images with shape (CxWxHxD).
    
    Original implementation taken from: 
        https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py
    """
    def __init__(self, random_state, **kwargs):
        """Rotate an array by 90 degrees around a randomly chosen plane.

        Args:
            random_state : 
        """
        self.random_state = random_state
        self.axis = (0, 1)

    def __call__(self, m):

        k = self.random_state.randint(0, 4)
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class Contrast:
    """Randomly changes the constrast of image with shape (CxWxHxD).
    
    Original implementation taken from: 
        https://github.com/MIC-DKFZ/batchgenerators/blob/de0a67d0f9916bb1bfdbe2aa77579774d2f2342b/batchgenerators/augmentations/color_augmentations.py#L22
    """
    def __init__(self, random_state, alpha=(0.75, 1.25),
                 execution_probability=0.5, **kwargs):
        """Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.

        Args:
            random_state :
            alpha (tuple, optional): range from which to sample a random contrast that is applied to the data. If
                                     one value is smaller and one is larger than 1, half of the contrast modifiers will be >1
                                     and the other half <1 (in the inverval that was specified). Defaults to (0.75, 1.25).
            execution_probability (float, optional): probability of applying mirroring. Defaults to 0.5.
        """
        self.random_state = random_state
        self.alpha = alpha
        self.exec_prob = execution_probability

    def __call__(self, m):
        mn = m[0].mean()
        minm = m[0].min()
        maxm = m[0].max()
        if self.random_state.uniform() < self.exec_prob and self.alpha[0] < 1:
            factor = np.random.uniform(self.alpha[0], 1)
        else:
            factor = np.random.uniform(max(self.alpha[0], 1), self.alpha[1])
        m[0] = (m[0] - mn) * factor + mn
        m[0][m[0] < minm] = minm
        m[0][m[0] > maxm] = maxm

        return m

class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes CWHD axis order.
    
    Implementation taken from:
        https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py
    """

    def __init__(self, random_state, spline_order=0, alpha=2000, sigma=50, execution_probability=0.1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            x_dim, y_dim, z_dim = volume_shape
            x, y, z = np.meshgrid(np.arange(x_dim), np.arange(y_dim), np.arange(z_dim), indexing='ij')
            indices = x + dx, y + dy, z + dz

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m


class Transformer:
    """For loading the augmentations from .json file and adding 
    them to the torch.tranforms.Compose.
    
    Original implementation taken from:
        https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py
    """
    def __init__(self, transform_list):
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)
        self.transform_list = transform_list
    
    @staticmethod
    def __transformer_class(class_name):
        m = importlib.import_module('dataset.transforms')
        return getattr(m, class_name)

    
    def create_transform(self):
        return Compose([self.__create_augmentation(c) for c in self.transform_list])

    def __create_augmentation(self, c):
        settings = dict()
        settings.update(c)
        settings['random_state'] = np.random.RandomState(self.seed)
        aug_class = self.__transformer_class(settings['name'])
        return aug_class(**settings)
