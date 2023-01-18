#MAIN TRAIN

import argparse
import pprint
import yaml
from core.model import *
from core.data import *
from core.trainer import *
from core.utils import *
import torch
import torch.multiprocessing as mp
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
     """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int=0, world_size: int=1, cfg=None):
    

    if cfg.cuda:
        ddp_setup(rank, world_size)
    
    train_data, val_data, model, optimizer = load_train_objs(cfg)
    
    trainer = Trainer(model, train_data, val_data, optimizer, cfg.save_every, cfg.total_epochs, cfg.logs_path, cfg.restart, cfg.cuda, rank)
    trainer.train()
    
    if cfg.cuda:
        destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-f','--config', type=str,
                          help='Configuration filepath that contains model parameters',
                          default = './config_file_multigpu.yaml')
    parser.add_argument('-j','--jobid', type=str,
                          help='JOB ID',
                          default = '000000')

    args = parser.parse_args()

    config_filepath = args.config

    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = dic2struc(cfg)
    
    if cfg.cuda:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args = (world_size, cfg), nprocs=world_size)
    else:
        main()
