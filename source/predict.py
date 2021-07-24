import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from unet import unet3d
from dataset.utils import *
from unet.utils import *
import numpy as np
import nibabel as nib
import argparse
from skimage import morphology
from nibabel.processing import resample_to_output, conform
from dataset.preprocessor import Preprocessor
from unet.trainer import UNetTrainerV2
from unet.metrics import DiceCoefficient

parser = argparse.ArgumentParser()
parser.add_argument('--input','-i',type=str)
parser.add_argument('--output','-o',type=str)
parser.add_argument('--model','-m',type=str)
parser.add_argument('--file','-f',type=str)
parser.add_argument('--eval','-e',action='store_true')

args = parser.parse_args()
preprocessor = Preprocessor(root_dir=args.input,
                            patch_size=(64,64,64),
                            crop=0,
                            pad_type='zero_pad',
                            look_for_labels=args.eval,
                            spacing=(0.8,0.8,0.8))
trainer = UNetTrainerV2(model_name=args.model,load=True)
metric = DiceCoefficient()

# preprocessing
data, label = preprocessor(args.file)
print("Preprocessing done.")
if args.eval:
    data = np.concatenate((np.expand_dims(data,axis=0),
                           np.expand_dims(label,axis=0)))
else: 
    data = np.expand_dims(data, axis=0)
# splitting to patches
data, unfold_shape = split_to_patches(data, (64,64,64), (64,64,64))
# inference
print("Starting inference...",end=' ')
results, seg_evals = trainer.infer(torch.tensor(data), metric)
print("Done.")
# rebuilding the image from patches
patches_orig = torch.tensor(results).view(unfold_shape)
output_c = unfold_shape[1] * unfold_shape[4]
output_h = unfold_shape[2] * unfold_shape[5]
output_w = unfold_shape[3] * unfold_shape[6]
patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
patches_orig = patches_orig.view(1, output_c, output_h, output_w)

if args.eval:
    seg_evals = np.array(seg_evals)
    seg_evals[seg_evals == 0.0] = np.nan
    print("Dice similarity coefficient is: {:.2%}".format(np.nanmean(seg_evals)))

image = patches_orig[0].cpu().numpy()

# saving image 
image = np.where((image >= 0.5) & (image < 1.1),255,0)
image = nib.Nifti1Image(image.astype(np.uint8), affine=np.eye(4))
nib.save(image,os.path.join(args.output, f"{args.model}_{args.file}"))

