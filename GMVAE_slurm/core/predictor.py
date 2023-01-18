# PREDICTOR

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
from core.model import GMVAE
from core.data import SchlierenDataModule
import numpy as np
import json

class Predictor():
   
    def __init__(self, cfg, snapshot_path = None):
       
        self.model = GMVAE(cfg)
        self.datamodule = SchlierenDataModule(cfg)
       
        if snapshot_path is None:
            self.snapshot_path = cfg.snapshot_path
        else:
            self.snapshot_path = snapshot_path
           
    def _load_inference_objs(self):
       
        #Load model
        snapshot = torch.load(self.snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        #Load prediction data
        self.datamodule.setup("predict")
       
    def predict(self):
       
        self._load_inference_objs()
        
        predict_set = self.datamodule.predict_dataloader()
                
        predictions = {}
        
        sample_count = 0
        for batch_id, data in enumerate(predict_set):

            source, targets = data
                        
            batch_size = source.shape[0]
            
            mu_z, logsigmasq_z, z, gamma_c, mu_x, logsigmasq_x, x_rec = self.model.predict(source)
            
            for sample in range(batch_size):
                predictions[f'sample_{sample}'] = {}
                predictions[f'sample_{sample}']['mu_z'] = mu_z.numpy()[sample,:].tolist()
                predictions[f'sample_{sample}']['logsigmasq_z'] = logsigmasq_z.numpy()[sample,:].tolist()
                predictions[f'sample_{sample}']['z'] = z.numpy()[sample,:].tolist()
                predictions[f'sample_{sample}']['gamma_c'] = gamma_c.numpy()[sample,:].tolist()
                predictions[f'sample_{sample}']['mu_x'] = mu_x.numpy()[sample,:,:].tolist()
                predictions[f'sample_{sample}']['logsigmasq_x'] = logsigmasq_x.numpy()[sample,:,:].tolist()
                predictions[f'sample_{sample}']['x_rec'] = x_rec.numpy()[sample,:,:].tolist()
                predictions[f'sample_{sample}']['label'] = np.argmax(gamma_c.numpy(), axis=1)[sample].tolist()
                predictions[f'sample_{sample}']['x'] = source.numpy()[sample,0,:,:].tolist()
            
                sample_count += 1
            
        return predictions

            
