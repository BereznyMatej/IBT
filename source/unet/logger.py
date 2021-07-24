"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    logger.py
Description:
    Implementation of loss and metric logging in every training process.
"""
import os
import time
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

class Logger:
    """Implements logging both as graphs and textfiles, preemptive training termination, saving best models.
    """
    def __init__(self, folder, net_name, json_dict):
        """Logger constructor.

        Args:
            folder (string): folder where to save best model versions.
            net_name (string): model name.
            json_dict (dict): dictionary with current model parameters loaded from .json file.
        """
        self.json_dict = json_dict  
        self.folder = folder
        self.name=net_name
        self.file = f"{self.name}_log_{json_dict['index']}.txt"
        self.path = os.path.join(self.folder,'Logs',self.file)
        self.mode = 'train'
        text = f"\nLog number {json_dict['index']} for self-suprevised UNet model {self.name}\n"
        
        self.epoch_time = None
        self.iter_time = None
        self.max_epoch = json_dict['max_epoch']
        self.train_dice = np.empty(json_dict['train_iter'])
        self.train_loss = np.empty(json_dict['train_iter'])
        self.valid_dice = np.empty(json_dict['val_iter'])
        self.valid_loss = np.empty(json_dict['val_iter'])
        self.patience = json_dict['patience']
        self.train_dice.fill(np.nan)
        self.train_loss.fill(np.nan)
        self.valid_dice.fill(np.nan)
        self.valid_loss.fill(np.nan)

        if not 'avg_train_loss' in self.json_dict:
            self.json_dict['avg_train_loss'] = []
        if not 'avg_valid_loss' in self.json_dict:
            self.json_dict['avg_valid_loss'] = []
        if not 'avg_train_dice' in self.json_dict:
            self.json_dict['avg_train_dice'] = []
        if not 'avg_valid_dice' in self.json_dict:
            self.json_dict['avg_valid_dice'] = []

        with open(self.path,'+a') as fp:
            fp.write(text)
        print(text,end='')
        self.json_dict['index']+=1  

    def update_metrics(self, index=0, clear=False, 
                       train_loss=None, train_dice=None,
                       valid_loss=None, valid_dice=None):
        """Updates loss,

        Args:
            index (int, optional): [description]. Defaults to 0.
            clear (bool, optional): If true, nulls current gathered losses and metrics. Defaults to False.
            train_loss (float, optional):  Defaults to None.
            train_dice (float, optional):  Defaults to None.
            valid_loss (float, optional):  Defaults to None.
            valid_dice (float, optional):  Defaults to None.
        """
        if clear:
            self.train_dice.fill(np.nan)
            self.train_loss.fill(np.nan)
            self.valid_dice.fill(np.nan)
            self.valid_loss.fill(np.nan)
            return
        
        if train_loss:
            self.train_dice[index] = train_dice
            self.train_loss[index] = train_loss
        elif valid_loss:
            self.valid_dice[index] = valid_dice
            self.valid_loss[index] = valid_loss

    def log_sample_size(self, train=None, valid=None):
        """Logs the number of patches present in of both training and validation subset.

        Args:
            train (int, optional): size of train subset. Defaults to None.
            valid (int, optional): size of validation subset. Defaults to None.
        """
        text = ""
        if train is not None:
            text += f"Total number of training samples is {len(train)}.\n"
        if valid is not None:
            text += f"Total number of validation samples is {len(valid)}."
        
        print(text)
        with open(self.path,'+a') as fp:
            fp.write(text)


    def time(self, iter=False):
        """Starts the timers for measuring epoch/iteration length.

        Args:
            iter (bool, optional): If True, iteration timer instead of epoch timer is started. Defaults to False.
        """
        if iter:
            self.iter_time = time.time()
        else:
            self.epoch_time = time.time()

    def iteration(self, epoch, iteration, verbose=False):
        """Logs the current average loss, metric and time it took to finish one iteration. 
        
        Args:
            epoch (int):
            iteration (int):
            verbose (bool, optional): If True,  logs the iteration.
        """
        
        text = """\n====Epoch [{}/{}], iteration {}, Loss: {:.6f}, Mean Dice: {:.2%}, Time: {:.2f}s""".format(
                          epoch + 1, self.json_dict['max_epoch'],
                          iteration + 1, np.nanmean(self.train_loss),
                          np.nanmean(self.train_dice),
                          time.time() - self.iter_time)
        if verbose:
            print(text,end='')
            with open(self.path,'+a') as fp:
                fp.write(text)
    
    def epoch(self, epoch, model_dict, optimizer_dict, verbose=True):
        """Logs the current average loss, metric and time it took to finish one epoch.
        Additionaly, function updates the model .json and .pt files and preemptively 
        terminates the training process, if no improvement was made in past 100 epochs. 

        """
        self.json_dict['epoch'] = epoch
        avg_train_loss = np.nanmean(self.train_loss)
        avg_train_dice = np.nanmean(self.train_dice)
        if self.mode == 'tune':
            avg_val_loss = avg_train_loss
            avg_val_dice = avg_train_dice
        else:
            avg_val_loss = np.nanmean(self.valid_loss)
            avg_val_dice = np.nanmean(self.valid_dice)
        text = """\nEpoch {}
        Training loss is {:.4f}, Validation loss is {:.4f},
        Mean Training Dice is {:.2%}, Mean Validation Dice is {:.2%},
        Time: {:.2f}s""".format(epoch+1,
                                avg_train_loss,
                                avg_val_loss,
                                avg_train_dice,
                                avg_val_dice,
                                time.time()-self.epoch_time)

        if avg_val_loss < self.json_dict['best_loss']:
            text += "\nValidation loss decreases from {:.4f} to {:.4f}".format(self.json_dict['best_loss'],
                                                                            avg_val_loss)
            self.json_dict['best_loss'] = avg_val_loss
            self.json_dict['epochs_no_improvement'] = 0
            text += "\nSaving model " + os.path.join(self.folder,f"{self.name}.pt")
            torch.save({
		        'epoch': epoch+1,
		        'state_dict' : model_dict,
		        'optimizer_state_dict': optimizer_dict
		    },os.path.join(self.folder, f"{self.name}.pt"))
        else:
            self.json_dict['epochs_no_improvement'] += 1
            text += """\nValidation loss does not decrease from {:.4f} epochs without improvement {}""".format(self.json_dict['best_loss'],
                                    self.json_dict['epochs_no_improvement'])
        if self.json_dict['epochs_no_improvement'] == self.patience:
            text += """\nEarly stopping since validation loss did not improve in past {} epochs""".format(self.patience)
        
        self.json_dict['avg_valid_loss'].append(avg_val_loss)
        self.json_dict['avg_train_loss'].append(avg_train_loss)
        self.json_dict['avg_train_dice'].append(avg_train_dice)
        self.json_dict['avg_valid_dice'].append(avg_val_dice)
        y_axis = np.arange(1,len(self.json_dict['avg_valid_loss'])+1)
        
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(4,4))
        ax.plot(y_axis, np.array(self.json_dict['avg_valid_loss']), c='r', label='Validation Losses')
        ax.plot(y_axis, np.array(self.json_dict['avg_train_loss']), c='b', label='Training Losses')
        ax.plot(y_axis, np.array(self.json_dict['avg_valid_dice']), c='g', label='Validation Dice')
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(os.path.join(self.folder,'Logs', f"{self.name}.pdf"))
        plt.close(fig)

        with open(os.path.join(self.folder,f"{self.name}.json"),"w") as fp:
                json.dump(self.json_dict, fp, sort_keys=True, indent=4)
        with open(self.path,'+a') as fp:
            fp.write(text)
        if verbose:
            print(text,end='')

        return self.json_dict['epochs_no_improvement'] == self.patience
