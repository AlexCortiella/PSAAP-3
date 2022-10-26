from core.model_VaDE import *
from core.data import *
from core.utilities import ProgressBar, Logger
import os

import yaml
import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
        
class dic2struc():
    def __init__(self, my_dict):
        if my_dict:
            for key in my_dict:
                setattr(self, key, my_dict[key])
        else:
            return None


def main(extra_args, data_args, model_args, train_args):

    seed_everything()


    #Setup data
    print('Data module initialized...\n')
    data_module = SurfReacDataModule(data_dir=data_args.data_dir,
                                     train_batch=data_args.train_batch,
                                     val_batch=data_args.val_batch,
                                     test_batch=data_args.test_batch,
                                     num_workers = data_args.num_workers
                                     )
                                     
    
    #Build model
    print('Model initialized...\n')

    model = VaDE(input_dim=model_args.input_dim,
                 latent_dim=model_args.latent_dim,
                 nClusters=model_args.nClusters,
                 inter_dims = model_args.inter_dims,
                 cuda = train_args.cuda
                 )
    
    
    #Callbacks
    callbacks = []
    if train_args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss')) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
    if train_args.enable_checkpointing:
        callbacks.append(ModelCheckpoint()) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html

    #Logger
    if train_args.logger:
        if train_args.default_root_dir is None:
            train_args.default_root_dir = os.getcwd()

        train_args.logger = Logger(save_dir=train_args.default_root_dir,
                                        default_hp_metric=False) #adding version=args['experiment'] will make the save directory the experiment name

    #Pytorch Lightning Trainer
    print('Trainer initialized...\n')
    if train_args.cuda:
        trainer = pl.Trainer(
            max_epochs = train_args.max_epochs,
            accelerator="gpu",
            strategy=train_args.strategy,
            devices=train_args.devices,
            num_nodes=train_args.num_nodes,
            default_root_dir=train_args.default_root_dir,
            callbacks=callbacks)
    else:
        trainer = pl.Trainer(
            max_epochs = train_args.max_epochs,
            accelerator="cpu",
            default_root_dir=train_args.default_root_dir,
            callbacks=callbacks)


    
    trainer.fit(model = model, datamodule=data_module)
    ckpt_dir = os.path.join(train_args.default_root_dir, "lightning_logs/last_checkpoint.ckpt")
    trainer.save_checkpoint(ckpt_dir)
    print("Training stage completed!")

    
    return data_module, model, trainer


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-f','--config', type=str,
                        help='Configuration filepath that contains model parameters',
                        default = './config_file_testmultigpu.yaml')
    parser.add_argument('-j','--jobid', type=str,
                        help='JOB ID',
                        default = '000000')

    args = parser.parse_args()

    config_filepath = args.config
    cfg = load_config(filepath=config_filepath)


    train_args = dic2struc(cfg['train'])
    model_args = dic2struc(cfg['model'])
    data_args = dic2struc(cfg['data'])
    extra_args = dic2struc(cfg['extra'])

    
    
            
    ##Call main program        
    data_module, model, trainer = main(extra_args, data_args, model_args, train_args)
