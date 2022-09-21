#This script contains the helper routines
import json
import cv2
import numpy as np
import torch
import os
import shutil

def denormalization(frame):
    mean, std = frame.mean(), frame.std()
    frame = frame * std + mean 
    minf = np.min(frame)
    maxf = np.max(frame)
    return ((frame - minf)/(maxf-minf)*255).astype(np.uint8)

def save_json(times_path, predictions, denormalize):
    
    steps = len(predictions["time"])
    frames = predictions["reconstructed"]
    latents = predictions["latent"]
    times = predictions["time"]

    my_dict = {}
    
    if not os.path.exists(times_path):
        os.makedirs(times_path, exist_ok=True)
    else:
        shutil.rmtree(times_path)
        os.makedirs(times_path, exist_ok=True)


    for i in range(steps):
        if type(frames[i]) == torch.Tensor:
            frame = frames[i].detach().numpy()
            latent = latents[i].detach().numpy()
            time = float(times[i].detach().numpy()[0])
        if denormalize:
            frame = denormalization(frame)
        if len(frame.shape) == 4:
            frame = frame[0,:,:,:]
        if len(latent.shape) == 2:
            latent = latent[0,:]
        if frame.shape[0] <= 3:
            frame = np.moveaxis(frame,0,-1)
            
            
        my_dict = {"reconstructed": frame.tolist(), "latent": latent.tolist(), "time":time, "metadata": predictions["metadata"]}
       
        filename = f'time_{time:.4f}.json'
        with open(os.path.join(times_path, filename), "w+") as outfile:
            print(f"Saving {filename} file!")
            json.dump(my_dict, outfile)
    

def write_video(filename, frames):
    
    size = list(frames[0].shape[2:])
    print(size)
    out = cv2.VideoWriter(f'{filename}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

    for frame in frames:
        if type(frame) == torch.Tensor:
            frame = frame.detach().numpy()
            
        frame = denormalize(frame)
        frame = np.moveaxis(frame,0,-1)
        out.write(frame)
    out.release()
    
def load_json(filename):
    # Opening JSON file
    with open(f'{filename}.json') as json_file:
        data_dict = json.load(json_file)
    return data_dict

def get_data(data_dict):
    
    frames = []
    latents = []
    for i in range(len(data_dict.keys())):
        frames.append(np.array(data_dict[f'step{i}']["reconstructed"]))
        latents.append(np.array(data_dict[f'step{i}']["latent"]))

    return frames, latents
