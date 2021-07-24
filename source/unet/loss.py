"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    loss.py
Description:
    Implementation of loss functions. 
"""
import torch
import torch.nn as nn
from torch import  Tensor
import torch.nn.functional as F
import numpy as np
from unet.utils import get_tp_fp_fn_tn


class DC_and_BCE_loss(nn.Module):
    """Combination of soft dice and binary cross entropy loss functions

     Original implementation from:
            https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
    """
    def __init__(self, bce_kwargs, soft_dice_kwargs):
        """Modified original implementation by removing application
        of nonlinear activation func. added support for sparse annotations. 

        Args:
            :param soft_dice_kwargs:
            :param bce_kwargs:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.aggregate = aggregate
        self.ce = F.binary_cross_entropy
        self.dc = SoftDiceLoss(**soft_dice_kwargs)

    def forward(self, net_output, target, mask=None):
        
        ce_loss = self.ce(net_output, target, mask)
        dc_loss = self.dc(net_output, target, mask)
        
        result = ce_loss + dc_loss
        
        return result

class SoftDiceLoss(nn.Module):
    """Soft dice loss.

    Implementation from: 
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """Soft dice loss for both sparse and dense annotations. Expects every input in format (BxCxWxHxD).

        Args:
            apply_nonlin (nonlinear operation, optional): If provided, nonlinear activation function such as 
                                                          sigmoid or softmax is applied. Defaults to None.
            batch_dice (bool, optional): compute dice loss across entire batch. Defaults to False.
            do_bg (bool, optional): Defaults to True.
            smooth (float, optional): Defaults to 1..
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc
