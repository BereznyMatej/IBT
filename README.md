Name: Matej Berezny
Login: xberez03
Thesis: Volumetric Segmentation of Dental CT Data

Set of scripts used for training 3D neural networks for medical image restoration task, or tooth segmentation from dental CT/CBCT scans. All scripts used for data manipulation are stored in `\dataset`, while scripts handling the network training are in `\unet`. 

### Preprocessing
To shape the data into the correct form before inputting them to the network, they were preprocessed by applying resampling, normalization, cropping and padding.
All preprocessing is handled in `preprocessor.py`, which is initiated automatically by launching script `preprocess.py`.
```sh 
python3 preprocess.py [-h] [--crop CROP] [--input INPUT] [--patch_size PATCH_SIZE] [--model MODEL] [--extrapolate]
                     [--no_label] [--spacing SPACING]
    
    '--help', '-h' Prints the short help message
    '--crop INT' '-c INT' sets the crop level:  '0' -> no cropping
                                                '1' -> crop around any tissue voxels
                                                '2' -> crop around bones
    '--patch_size INT' '--p INT' number specifying isotropic shape of patches (must be multiple of 16)
    '--model STRING' '-m STRING' name under which will be the preprocessed dataset saved in '\preprocessed'
    '--extrapolate' '-e' extrapolate the images to match padding level, 
                         if not specified, 'zero-pad'  will be used instead
    '--no_label' '-n' ignores any '\labels' folders, preprocessed data for self-supervised task
    '--spacing INT' '-s INT' number specifying isotropic target spacing for resampling
```

## Training
Training process last for set number of epochs with best versions of model saved in `\pretrained weights`. If no improvement is being made in past 100 epochs, training gets terminated preemptively.
All training is handled by `UNetTrainer` and `Logger`, which are initiated automatically by launching script `train.py`.
If user wishes to further modify the training environment besides the options provided by command line params and 
.json files, (such as changing loss function, evaluation metric) `train.py` or `UNetTrainer` have to be modified.
```sh 
python3 train.py [-h] [--model MODEL] [--data DATA] [--load_weights] [--skip_validation] [--sparse_ann]
    
    '--help', '-h' Prints the short help message
    '--model STRING' '-m STRING' specifies the name under which will be the model saved in '\pretrained_weights'
    '--data STRING' '-d STRING' name of the preprocessed dataset
    '--load_weights' '-l' If set, loads the model weights from '\pretrained_weights' folder.
    '--skip_validation' If set, validation part of training will be skipped, 
                        best loss will be determined from training instead.
    '--sparse_ann' If set, weight masks will be generated for each of the label volumes.
```
## Inference
Runs inference on one file from directory set in `--input`. Both have to be specified, along with the model name. If user wishes to also calculate dice coeff., label has to be provided in `'{INPUT_DIR}\labels'`

```sh
python3 predict.py [-h] [--input INPUT] [--output OUTPUT] [--model MODEL] [--file FILE] [--eval]

    '--help', '-h' Prints the short help message
    '--input DIR' '-i DIR' directory containing input files.
    '--output DIR' '-o DIR' where to save finished predictions.
    '--file FILE' '-f FILE' name of the file from 'input' dir for running inference on.
    '--eval' '-e' calculates dice similarity coefficient between result of inference and supplied label.
                  Requires label to be in '{INPUT_DIR}\labels' directory.
    '--model STRING' '-m STRING' specifies the model used for inference. 
```

# Pretrained models
Directory `\pretrained_weights` contains most of the models used in thesis's experiments. best versions of models are marked as `BEST_{model_name}`.