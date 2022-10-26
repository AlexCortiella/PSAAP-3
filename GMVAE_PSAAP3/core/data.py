import urllib.parse
import json
import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

#### DATA LOADER ####

class SurfReacDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./", n_train: int = 5000, n_val: int = 100, n_test: int = 100, train_batch: int = 1, val_batch: int = 1, test_batch: int = 1, num_workers = 8):
        super().__init__()
        self.data_dir = data_dir
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.n_samples = n_train + n_val + n_test
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        self.num_workers = num_workers


    def setup(self, stage = None):
        
        
        f = open(self.data_dir)
        data_dict = json.load(f)
        
        assert(self.n_samples <= len(data_dict))

        data_tr = []
        label_tr = []
        data_val = []
        label_val = []
        data_test = []
        label_test = []
        
        for i in range(self.n_samples):

            if i>=0 and i < self.n_train:
                data_tr.append(torch.tensor(data_dict[f'sample{i+1}']['data'], dtype = torch.float))
                label_tr.append(torch.tensor(data_dict[f'sample{i+1}']['label'], dtype = torch.uint8))
            elif i>=self.n_train and i < self.n_train + self.n_val:
                data_val.append(torch.tensor(data_dict[f'sample{i+1}']['data'], dtype = torch.float))
                label_val.append(torch.tensor(data_dict[f'sample{i+1}']['label'], dtype = torch.uint8))
            else:
                data_test.append(torch.tensor(data_dict[f'sample{i+1}']['data'], dtype = torch.float))
                label_test.append(torch.tensor(data_dict[f'sample{i+1}']['label'], dtype = torch.uint8))        

        self.train_dataset = tuple(zip(data_tr, label_tr))
        self.val_dataset = tuple(zip(data_val, label_val))
        self.test_dataset = tuple(zip(data_test, label_test))
        
        self.input_dim = len(self.train_dataset[0][0])

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_stage_dataset =  self.train_dataset
            self.val_stage_dataset = self.val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_stage_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch, shuffle=False, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch, shuffle=False, num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch, shuffle=False, num_workers = self.num_workers)
