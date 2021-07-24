"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    metrics.py
Description:
    Implementation of dice coefficient evaluation function.
"""
import numpy as np
import torch
import torch.nn.functional as F
from unet.utils import flatten

class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    
    Implementation originally from: 
        https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/metrics.py
    
    Extended by implementing correct dice calculations in case of sparse annotations.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target, weights=None):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        if weights is not None:
            input = input[weights > 0.5]
            target = target[weights > 0.5]
        else:
            input = flatten(input)
            target = flatten(target)
        target = target.float()
        
        intersect = (input * target).sum(-1)
        denominator = (input**2 + target**2).sum(-1)
        return 2 * (intersect / denominator.clamp(min=self.epsilon))
