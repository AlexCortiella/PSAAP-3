#This is the main script for the Schlieren Autoencoder

import pytorch_lightning as pl
import yaml

from core.data import getDataset
from core.model import CNNAutoencoder
from utilities.utils import save_json

from torchsummary import summary
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
    #config_filepath = str(sys.argv[1])
    config_filepath = 'config_file.yaml'
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = dic2struc(cfg)

    #Data module
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
    my_model = CNNAutoencoder()

    #Train model
    if cfg.if_cuda:
        trainer = pl.Trainer(max_epochs = cfg.epochs, accelerator="gpu", strategy="ddp", devices=cfg.num_devices, num_nodes=cfg.num_nodes)
    else:
        trainer = pl.Trainer(max_epochs = cfg.epochs, accelerator="cpu", strategy="ddp", devices=cfg.num_devices, num_nodes=cfg.num_nodes)

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
    predictions["metadata"] = {'fps': 1/cfg.step, 'start': cfg.start, 'stop': cfg.stop, 'height': cfg.height, 'width':cfg.width}    
    for (rec, lat) in preds:
        predictions["latent"].append(lat)
        predictions["reconstructed"].append(rec)

    #Save results
    print("Saving results to json file...")
    save_json(cfg.model_name, predictions, True)
    print("Saved!")
    
if __name__ == '__main__':
    
    main()
    

