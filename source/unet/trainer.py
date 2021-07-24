"""
Thesis:
    Volumetric Segmentation of Dental CT Data
Author:
    Matej Berezny
File:
    trainer.py
Description:
    Implementation of trainer class.
"""
import numpy as np 
import torch 
import sys
import torch.nn as nn
from unet.unet3d import UNet3D

class UNetTrainer:
    def __init__(self, start_epoch=0, end_epoch=1000, 
                 criterion=None, metric=None, logger=None,
                 model_name="", load=False, step_size=int(50 * 0.8),
                 learning_rate=0.01, mode='train'):
        """Initializes the UNet3D model, Adam optimizer, scheduler.

        Args:
            start_epoch (int, optional):  Defaults to 0.
            end_epoch (int, optional):  Defaults to 1000.
            criterion (loss, optional): loss fucntion. Defaults to None.
            metric (eval, optional): evaluation function. Defaults to None.
            logger (Logger, optional): Defaults to None.
            model_name (str, optional): Defaults to "".
            load (bool, optional): load weights from file. Defaults to False.
            step_size (int, optional): step size for scheduler. Defaults to int(50 * 0.8).
            learning_rate (float, optional): learning rate for adam optimizer. Defaults to 0.01.
            mode (str, optional): 'train' normally, 'tune' for skipping validation. Defaults to 'train'.
        """
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.metric = metric
        self.logger = logger
        self.mode = mode 
        if self.logger is not None:
            self.logger.mode = mode  
        self.model = UNet3D()
        self.device=torch.device('cuda' if torch.cuda.is_available() else  'cpu')
        sys.stdout.flush()

        if load:
            checkpoint = torch.load(f"../pretrained_weights/{model_name}.pt")
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            self.model.load_state_dict(unParalled_state_dict)

        self.model = nn.DataParallel(self.model, device_ids = [i for i in range(torch.cuda.device_count())])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=step_size,
                                                         gamma=0.5)
        
    def fit(self, train_data=None, valid_data=None):
        """Initiates the training cycle.

        Args:
            train_data (DataLoader, optional): dataloader with training subset. Defaults to None.
            valid_data (DataLoader, optional): dataloader with validation subset. Defaults to None.
        """
        for epoch in range(self.start_epoch, self.end_epoch):
            # training part
            self.train(train_data, epoch)
            # validation part
            if self.mode != 'tune':
                self.validate(valid_data, epoch)
            self.scheduler.step()
            # logging the epoch details
            if self.logger.epoch(epoch, self.model.state_dict(), self.optimizer.state_dict()):
                print("Early Stopping")
                break
            # clear the loss and metric array
            self.logger.update_metrics(clear=True)
            sys.stdout.flush()
    
    def train(self, train_loader, epoch):
        """Trains the network for one epoch, with precomputed masks.

        Args:
            train_loader (DataLoader): dataloader with training subset. Defaults to None.
            epoch (int): current epoch number.
        """
        self.model.train()
        self.logger.time()
        for index, patch in enumerate(train_loader):
            self.logger.time(iter=True)
            x, y = patch['x'].float().to(self.device), patch['y'].float().to(self.device)
            pred=self.model(x)
            mask = patch['mask'].float().to(self.device)
            loss = self.criterion(pred, y, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            self.logger.update_metrics(index=index, train_loss=round(loss.item(), 2),
                              train_dice=self.metric(pred,y,mask))
            if (index + 1) % 5 == 0:
                self.logger.iteration(epoch, index, False)
                sys.stdout.flush()


    def validate(self, valid_loader, epoch):
        """Performs evaluation on entire validation subset, with precomputed masks.

        Args:
            valid_loader (DataLoader): dataloader with validation subset. Defaults to None.
            epoch (int): current epoch number.
        """
        with torch.no_grad():
            self.model.eval()
            for index, patch in enumerate(valid_loader):
                x, y = patch['x'].float().to(self.device), patch['y'].float().to(self.device)
                pred=self.model(x)
                mask = patch['mask'].float().to(self.device)
                loss = self.criterion(pred, y, mask)
                
                self.logger.update_metrics(index=index, valid_loss=loss.item(),
                                      valid_dice=self.metric(pred,y,mask))

    def infer(self, data, metric):
        """Performs inference on given data, calculates dice coeff. if annotations 
        are provided.

        Args:
            data (tensor): data to be infered. Shape: (BxCxHxWxD)
            metric (DiceCoefficient): 

        Returns:
            np.array: network predictions
            list: list with dice results.
        """
        results = []
        seg_evals = []
        with torch.no_grad():
            self.model.eval()
            for index in range(data.shape[0]):
                patch = torch.unsqueeze(data[index],dim=1)
                x = torch.unsqueeze(patch[0],dim=0).float().to(self.device)
                pred=self.model(x)
                results.append(pred.cpu().numpy())
                if patch.shape[0] == 2:
                    y = torch.unsqueeze(patch[1],dim=0).float().to(self.device)
                    seg_evals.append(metric(pred,y).item())
            return np.concatenate(results), seg_evals
            
class UNetTrainerV2(UNetTrainer):
    def train(self, train_loader, epoch):
        """Trains the network for one epoch, without precomputed masks.

        Args:
            train_loader (DataLoader): dataloader with training subset. Defaults to None.
            epoch (int): current epoch number.
        """
        self.model.train()
        self.logger.time()
        for index, patch in enumerate(train_loader):
            self.logger.time(iter=True)
            x, y = patch['x'].float().to(self.device), patch['y'].float().to(self.device)
            pred=self.model(x)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            self.logger.update_metrics(index=index, train_loss=round(loss.item(), 2),
                              train_dice=self.metric(pred,y))
            if (index + 1) % 5 == 0:
                self.logger.iteration(epoch, index, False)
                sys.stdout.flush()

    def validate(self, valid_loader, epoch):
        """Performs evaluation on entire validation subset, without precomputed masks.

        Args:
            valid_loader (DataLoader): dataloader with validation subset. Defaults to None.
            epoch (int): current epoch number.
        """
        with torch.no_grad():
            self.model.eval()
            for index, patch in enumerate(valid_loader):
                x, y = patch['x'].float().to(self.device), patch['y'].float().to(self.device)
                pred=self.model(x)
                loss = self.criterion(pred, y)
                pred=self.model(x)
                
                self.logger.update_metrics(index=index, valid_loss=loss.item(),
                                      valid_dice=self.metric(pred,y))
