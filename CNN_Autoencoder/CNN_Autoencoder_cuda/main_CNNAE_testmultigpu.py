#This is the main script for the Schlieren Autoencoder

import pytorch_lightning as pl
import yaml
import os
import argparse

from core.data import getDataset
from core.model import CNNAutoencoder
from utilities.utils import save_json

from torchinfo import summary

import cv2
import matplotlib.pyplot as plt
import pprint

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


def main():
    
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
    pprint.pprint(cfg)
    cfg = dic2struc(cfg)

    model_name = f'{cfg.model_name}_JOBID_{args.jobid}'
    results_path = os.path.join(cfg.out_dir, model_name)

    #Data module
    print('Instantiating data module...')
    my_datamodule = getDataset(
                    train_dir=cfg.train_dir,
                    val_dir = cfg.val_dir,
                    test_dir = cfg.test_dir,
                    pred_dir = cfg.pred_dir,
                    start = cfg.start,
                    stop = cfg.stop,
                    step = cfg.step,
                    height = cfg.height, 
                    width = cfg.width,
                    num_workers = cfg.num_workers,
                    batch_size = cfg.train_batch)

    #Create an instance of the CNNae model 
    hdiff, wdiff = int(abs(cfg.height[1] - cfg.height[0])), int(abs(cfg.width[1] - cfg.width[0]))
    idims = (1, hdiff, wdiff)
    print('Instantiating model  module...')
    my_model = CNNAutoencoder(dims=idims, layers = cfg.cnn_layers, fc_layers = cfg.fc_layers, max_channels = cfg.max_channels, kernel_size = cfg.kernel_size)

    #Train model
    print('Instantiating train module...')
    log_dir = results_path

    if cfg.if_cuda:
        trainer = pl.Trainer(max_epochs = cfg.epochs, accelerator="gpu", strategy="ddp", devices=cfg.num_devices, num_nodes=cfg.num_nodes, default_root_dir=log_dir)
    else:
        trainer = pl.Trainer(max_epochs = cfg.epochs, accelerator="cpu", strategy="ddp", devices=cfg.num_devices, num_nodes=cfg.num_nodes, default_log_dir=log_dir)

    print("Initializing training stage...")
    trainer.fit(model = my_model, datamodule = my_datamodule)
    print("Training stage completed!")

    #Predict
    print("Initializing prediction stage...") 
    preds = trainer.predict(model = my_model, datamodule = my_datamodule)
    print("Prediction stage completed!")
    
    predictions = {}
    predictions["latent"] = []
    predictions["reconstructed"] = []
    predictions["time"] = []
    predictions["metadata"] = {'fps': 1/cfg.step, 'start': cfg.start, 'stop': cfg.stop, 'height': cfg.height, 'width':cfg.width}    
    for (rec, lat, time) in preds:
        predictions["latent"].append(lat)
        predictions["reconstructed"].append(rec)
        predictions["time"].append(time)

    #Save results
    print("Saving results to json file...")
    times_path = os.path.join(results_path, 'times')
    save_json(times_path, predictions, True)
    print("Saved!")

    print("Saving configuration file...")
    with open(os.path.join(results_path,f'{model_name}.yaml'), 'w+') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    
if __name__ == '__main__':
    
    main()
   
