import numpy as np
import json
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
#from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
import os
import cv2


class getVideoData(Dataset):
    
    def __init__(self, data_dir, normalize = True):
        
        self.data_dir = data_dir
        self.current_file = self.open_video_file()
        self.normalize = normalize
        
    def open_video_file(self, filename = None):
        
        if filename is None:
            filepath = self.data_dir
        else:
            filepath = os.path.join(self.data_dir, filename)
            
        try:
            video_file = cv2.VideoCapture(filepath)
            self.current_file = video_file
            
        except FileNotFoundError:
            print("Wrong file or file path")
        
        
        
        self.file_info = self.get_file_info()
            
        return video_file
    
    def get_file_info(self, verbose = True):
        
        file_info = {}
        
        if verbose:
            print('VIDEO FILE PROPERTIES')
            print('---------------------')
            # showing values of the properties
            print("Frame Width (px): {}".format(int(self.current_file.get(cv2.CAP_PROP_FRAME_WIDTH))))
            print("Frame Height (px): {}".format(int(self.current_file.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            print("Fps: '{}'".format(int(self.current_file.get(cv2.CAP_PROP_FPS))))
            print("Total number of frames: {}".format(int(self.current_file.get(cv2.CAP_PROP_FRAME_COUNT))))
            print("Video duration: {} sec.".format(int(self.current_file.get(cv2.CAP_PROP_FRAME_COUNT))/int(self.current_file.get(cv2.CAP_PROP_FPS))))
            print("Current Position (ms): {}".format(self.current_file.get(cv2.CAP_PROP_POS_MSEC)))
            print("Brightness: {}".format(self.current_file.get(cv2.CAP_PROP_BRIGHTNESS)))
            print("Contrast: {}".format(self.current_file.get(cv2.CAP_PROP_CONTRAST)))
            print("Saturation Value: {}".format(self.current_file.get(cv2.CAP_PROP_SATURATION)))
            print("HUE Value: {}".format(self.current_file.get(cv2.CAP_PROP_HUE)))
            print("Gain: {}".format(self.current_file.get(cv2.CAP_PROP_GAIN)))
        
        file_info['FrameWidth'] = int(self.current_file.get(cv2.CAP_PROP_FRAME_WIDTH))
        file_info['FrameHeight'] = int(self.current_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file_info['Fps'] = int(self.current_file.get(cv2.CAP_PROP_FPS))
        file_info['CurrentPosition'] = self.current_file.get(cv2.CAP_PROP_POS_MSEC)
        file_info['NumberOfFrames'] = int(self.current_file.get(cv2.CAP_PROP_FRAME_COUNT))
        file_info['VideoDuration'] = int(self.current_file.get(cv2.CAP_PROP_FRAME_COUNT))/int(self.current_file.get(cv2.CAP_PROP_FPS))
        file_info['Brightness'] = self.current_file.get(cv2.CAP_PROP_BRIGHTNESS)
        file_info['Contrast'] = self.current_file.get(cv2.CAP_PROP_CONTRAST)
        file_info['SaturationValue'] = self.current_file.get(cv2.CAP_PROP_SATURATION)
        file_info['HUEValue'] = self.current_file.get(cv2.CAP_PROP_HUE)
        file_info['Gain'] = self.current_file.get(cv2.CAP_PROP_GAIN)
        
        return file_info
        
    def process_video(self, start = None, stop = None, step = None, height = None, width = None):
        
        fps = self.file_info['Fps']
        
        if start is None: start = 0.0
        if ((stop is None) or (stop > self.file_info['VideoDuration'])): stop = self.file_info['VideoDuration']
        if step is None: step = 1./fps
        rem = (step%(1./fps))
        if rem != 0.0: step = step - rem
        if height is None: height = [0, self.file_info['FrameHeight']]
        if width is None: width = [0,self.file_info['FrameWidth']]
            
        assert(start < stop)
        assert(step >= 1.0/fps)
        
        start_frame = round(start * fps)
        stop_frame = round(stop * fps)
        step_frame = round(step * fps)
                
        #Start frame
        data_frames = []
        time_stamps = []
        
        for i in range(start_frame, stop_frame, step_frame):
            self.current_file.set(1,i)
            ret, frame = self.current_file.read()
            #[frames, channels, height, width] Pytorch format
            # Since the video is in grayscale, we remove the channel axis
            frame_mod = frame[height[0]:height[-1], width[0]:width[-1],0:1]
            
            if self.normalize:
                mean, std = frame_mod.mean(), frame_mod.std()
                frame_mod = (frame_mod - mean) / std
            data_frames.append(torch.from_numpy(np.moveaxis(frame_mod, -1, 0)).float())
            time_stamps.append(torch.tensor(i / fps).float())
        
        return list(zip(data_frames, time_stamps))
    
    def __getitem__(self, filename):
        
        video_file = self.open_video_file(filename)
        data, time = self._process_video(video_file)
            
        return data

class getDataset(LightningDataModule):
    def __init__(self,
        train_dir : str,
        val_dir : str, 
        test_dir : str = None, 
        pred_dir : str = None, 
        shuffle = True, 
        batch_size = 1, 
        num_workers = 1, 
        start = None,
        stop = None,
        step = None,
        height = None,
        width = None,
        **kwargs):

        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.pred_dir = pred_dir
        self.kwargs = kwargs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_trajectories = []
        self.val_trajectories = []
        self.test_trajectories = []
        self.pred_trajectories = []
        
        self.height = height
        self.width = width
        self.start = start
        self.stop = stop
        self.step = step
        
    #Loop over folders containing the different train, val, and test simulations (esnsemble of trajectories)
    def setup(self, stage = None):
            
        if stage in (None, 'fit'):
            train_dirs = [os.path.join(self.train_dir, d) for d in os.listdir(self.train_dir)] 
            
            for d in train_dirs:
                print(f'Processing training example {d.split("/")[-1]} out of {len(train_dirs)}') 
                video = getVideoData(d)
                trajectory = video.process_video(start = self.start, stop = self.stop, step = self.step, height = self.height, width = self.width)
                self.train_trajectories.append(trajectory)
                
            if self.val_dir is not None:
                val_dirs = [os.path.join(self.val_dir, d) for d in os.listdir(self.val_dir)] 
            else:
                val_dirs = []
                
            for d in val_dirs:
                print(f'Processing validation example {d.split("/")[-1]} out of {len(val_dirs)}')
                video = getVideoData(d)
                trajectory = video.process_video(start = self.start, stop = self.stop, step = self.step, height = self.height, width = self.width)
                self.val_trajectories.append(trajectory)

        if stage in (None, 'test'):
            if self.test_dir is not None:
                test_dirs = [os.path.join(self.test_dir, d) for d in os.listdir(self.test_dir)] 
            else:
                test_dirs = []

            for d in test_dirs:
                print(f'Processing test example {d.split("/")[-1]} out of {len(test_dirs)}')
                video = getVideoData(d)
                trajectory = video.process_video(start = self.start, stop = self.stop, step = self.step, height = self.height, width = self.width)
                self.test_trajectories.append(trajectory)
            
        if stage in (None, 'predict'):
            if self.pred_dir is not None:
                pred_dirs = [os.path.join(self.pred_dir, d) for d in os.listdir(self.pred_dir)] 
            else:
                pred_dirs = [] 
            
            for d in pred_dirs:
                print(f'Processing prediction example {d.split("/")[-1]} out of {len(pred_dirs)}')
                video = getVideoData(d)
                trajectory = video.process_video(start = self.start, stop = self.stop, step = self.step, height = self.height, width = self.width)
                self.pred_trajectories.append(trajectory)
                
    def train_dataloader(self):
        print('Creating training dataloader...')
        dataset = ConcatDataset(self.train_trajectories)##Concatenates the ensemble of trajectories
#         dataset = self.train_trajectories##Concatenates the ensemble of trajectories
        
        return DataLoader(
            dataset = dataset, 
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
        )

    def test_dataloader(self):
        print('Creating test dataloader...')
        dataset = ConcatDataset(self.test_trajectories)
        return DataLoader(
            dataset = dataset, 
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
        )
    def val_dataloader(self):
        print('Creating validation dataloader...')
        dataset = ConcatDataset(self.val_trajectories)
        return DataLoader(
            dataset = dataset, 
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
        )
        
    def predict_dataloader(self):
        print('Creating prediction dataloader...')
        dataset = ConcatDataset(self.pred_trajectories)
        return DataLoader(
            dataset = dataset, 
            batch_size = 1,
            shuffle = False,
            num_workers = self.num_workers,
        )
