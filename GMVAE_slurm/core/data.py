# DATA

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

import pandas as pd
import numpy as np
import os


## Data loader

class SchlierenDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = lambda x : -1 + 2*x
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, torchvision.io.ImageReadMode.GRAY).to(torch.float32) / 255.
        
        label = torch.tensor(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class SchlierenDataModule():
    def __init__(self, cfg):
        super().__init__()
        self.train_dir = cfg.train_dir
        self.pred_dir = cfg.pred_dir
        self.batch_size = cfg.batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.cuda = cfg.cuda

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            annotations_dir = os.path.join(self.train_dir, 'annotations')
            data_dir = os.path.join(self.train_dir, 'images')
            schlieren_full = SchlierenDataset(annotations_dir + "/annotations_file.csv", data_dir)
            n_fit = len(schlieren_full)
            n_train = int(0.9*n_fit)
            n_val = n_fit - n_train
            self.schlieren_train, self.schlieren_val = random_split(schlieren_full, [n_train, n_val])
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
#             self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            pass
        if stage == "predict":
            annotations_dir = os.path.join(self.pred_dir, 'annotations')
            data_dir = os.path.join(self.pred_dir, 'images')
            self.schlieren_predict = SchlierenDataset(annotations_dir + "/annotations_file.csv", data_dir)
    def train_dataloader(self):
        
        if self.cuda:
            sampler = DistributedSampler(self.schlieren_train)
        else:
            sampler = None
            
        return DataLoader(self.schlieren_train,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          sampler = sampler)
         
    def val_dataloader(self):
        
        
        if self.cuda:
            sampler = DistributedSampler(self.schlieren_val)
        else:
            sampler = None
            
        return DataLoader(self.schlieren_val,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          sampler=sampler)
    

    def test_dataloader(self):
        return DataLoader(self.schlieren_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        
        return DataLoader(self.schlieren_predict, batch_size=self.batch_size)
