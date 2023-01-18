#UTILITIES
import yaml
import torch
from core.data import SchlierenDataModule
from core.model import GMVAE
from torch.distributed import init_process_group
import os

#Configuration file reader utilities
class dic2struc():
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

        
#Pytorch distributed DDP utilities

def ddp_setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    init_process_group(backend="nccl", rank = rank, world_size = world_size)
    
def load_train_objs(cfg):
    
    #Prepare data
    datamodule = SchlierenDataModule(cfg)
    datamodule.setup('fit')# load your dataset
    
    train_set = datamodule.train_dataloader()#train DataLoader   
    val_set = datamodule.val_dataloader() #val DataLoader

    #Instantiate model
    model = GMVAE(cfg)  # load your model

    #IConfigure optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    return train_set, val_set, model, optimizer
