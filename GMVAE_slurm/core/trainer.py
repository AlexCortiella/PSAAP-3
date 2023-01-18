# TRAINER
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tensorboard_logger import configure, log_value
import os

class Trainer:
    def __init__(
        self,
        model,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        total_epochs: int,
        logs_path: str,
        restart: bool = False,
        cuda: bool = False,
        gpu_id: int = 0
    ) -> None:
        
        if cuda:
            self.gpu_id = gpu_id       
            self.model = model.to(self.gpu_id)
            ddp_model = DDP(self.model, device_ids=[self.gpu_id])
            self.model = ddp_model.module
        else:
            self.model = model
       
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.total_epochs = total_epochs
        self.epochs_run = 0
    
        self.logs_path = logs_path
        self.restart = restart
        self.run_path = None
        self.current_run = 0
        
        self.cuda = cuda        
        
        self.train_loss = []
        self.val_loss = []
        self.pi_history = np.expand_dims(np.zeros_like(self.model.params['pi_c'].detach().clone()), axis = 1)
        
        self.mu_c_history = np.expand_dims(np.zeros_like(self.model.params['mu_c']), axis = 2)
        self.logsigmasq_c_history = np.expand_dims(np.zeros_like(self.model.params['logsigmasq_c']), axis = 2)
        self.min_val_loss = torch.inf
        
        self._check_run()
        
        configure(self.run_path)
            
    def _check_run(self):
        
        logs_path = self.logs_path
        
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
            
        run_list = [item for item in os.listdir(logs_path) if item.split('_')[0] == 'run']
        if run_list:
            idx = [int(item.split('_')[-1]) for item in run_list]
            idx.sort()
            last_run = idx[-1]
            
            #If the epochs ran in the last run are less than the total epochs then load the model and restart
            snapshot_path = os.path.join(logs_path, f'run_{last_run}/snapshot.pt')
            if self.restart and os.path.exists(snapshot_path):
                
                if self.cuda:
                    loc = f"cuda:{self.gpu_id}"
                    snapshot = torch.load(snapshot_path, map_location=loc)
                else:
                    snapshot = torch.load(snapshot_path)
                
                if snapshot['EPOCHS_RUN'] < snapshot['TOTAL_EPOCHS']-1:
                    print("Loading snapshot")
                    self.current_run = last_run
                    self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
                    self._load_snapshot(snapshot_path)
                    self._load_variables(os.path.join(logs_path, f'run_{last_run}'))
                else:
                    self.current_run = last_run + 1
                    self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
                    os.makedirs(self.run_path)
            else:
                self.current_run = last_run + 1
                self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
                os.makedirs(self.run_path)            
        else:
            self.current_run = 0
            self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
            os.makedirs(self.run_path)   

    def _load_snapshot(self, snapshot_path):
        if self.cuda:
            loc = f"cuda:{self.gpu_id}"
            snapshot = torch.load(snapshot_path, map_location=loc)
        else:
            snapshot = torch.load(snapshot_path)
            
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.model.eval()
        self.epochs_run = snapshot["EPOCHS_RUN"]
        
        print(f"Resuming training from run_{self.current_run} snapshot at Epoch {self.epochs_run}")
        
    def _load_variables(self, variables_path):
        
        self.pi_history = np.load(os.path.join(variables_path,'pi_epoch.npy'))
        self.mu_c_history = np.load(os.path.join(variables_path,'mu_c_epoch.npy'))
        self.logsigmasq_c_history = np.load(os.path.join(variables_path,'logsigmasq_c_epoch.npy'))

    def _run_batch_train(self, source, targets):
        
        self.optimizer.zero_grad()
        
        loss = self.model.loss(source)
        print("Loss computed") 
        loss.backward()
        print("Backward pass completed")
        self.optimizer.step()
        print("Optimization step applied") 
        return loss
    
    def _run_batch_val(self, source, targets):
                
        loss = self.model.loss(source)
                        
        return loss

    def _run_epoch(self, epoch, dataset, mode='train'):
        
        b_sz = len(next(iter(dataset))[0])
        if self.cuda:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(dataset)}")
            dataset.sampler.set_epoch(epoch)
        else:
            print(f"Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(dataset)}")

        for step, data in enumerate(dataset):

            source, targets = data
            print("Extracted source and targets") 
            if self.cuda:
                source = source.to(self.gpu_id)
                print(f"Source shape: {source.shape}")
                print(f"Source sent to GPU{self.gpu_id}")
                targets = targets.to(self.gpu_id)
                print(f"Targets sent to GPU{self.gpu_id}")
            if mode == 'train':
                loss = self._run_batch_train(source, targets)
                print("Loss computed")
            else:
                loss = self._run_batch_val(source, targets)
                
            print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item()}")
            
        return loss
                
    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            "TOTAL_EPOCHS": self.total_epochs,
        }
        snapshot_path = os.path.join(self.run_path, 'snapshot.pt')
        torch.save(snapshot, snapshot_path)
        print(f"\nEpoch {epoch} | Training snapshot saved at {snapshot_path}")

    def train(self):
        max_epochs = self.total_epochs
        for epoch in range(self.epochs_run, max_epochs):
            
            #Reset running weights
            self.model.params['hist_weights'] = torch.zeros((self.model.n_clusters, 1))
            self.model.params['hist_mu_c'] = torch.zeros_like(self.model.params['mu_c'])
            self.model.params['hist_logsigmasq_c'] = torch.zeros_like(self.model.params['logsigmasq_c'])
            
            #Training
            print('\nTraining:')
            train_loss = self._run_epoch(epoch, self.train_data, 'train')
            self.train_loss.append(train_loss.item())
            log_value('train_loss_epoch', train_loss.item(), epoch)
            
            self.pi_history = np.concatenate((self.pi_history,torch.unsqueeze(self.model.params['pi_c'].detach().clone(), dim=1)), axis = 1)
            self.mu_c_history = np.concatenate((self.mu_c_history,np.expand_dims(self.model.params['mu_c'].detach(), axis = 2)), axis = 2)
            self.logsigmasq_c_history = np.concatenate((self.logsigmasq_c_history,np.expand_dims(self.model.params['logsigmasq_c'].detach(), axis = 2)), axis = 2)

            np.save(self.run_path + '/pi_epoch',self.pi_history)
            np.save(self.run_path + '/mu_c_epoch',self.mu_c_history)
            np.save(self.run_path + '/logsigmasq_c_epoch',self.logsigmasq_c_history)
            
            if self.cuda:
                if self.gpu_id == 0 and epoch % self.save_every == 0:
                    self._save_snapshot(epoch)
            else:
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch)

            #Validation
            with torch.no_grad():
                print('\nValidation:')
                val_loss = self._run_epoch(epoch, self.val_data, 'val')
                self.val_loss.append(val_loss.item())
                log_value('val_loss_epoch', val_loss.item(), epoch)
            
