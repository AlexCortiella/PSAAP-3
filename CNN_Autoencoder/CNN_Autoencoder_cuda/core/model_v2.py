import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt

class ResNetBlock(nn.Module):
    r""" A residual network block

    """
    def __init__(self, chan, kernel_size = 3, activation = torch.nn.LeakyReLU):
        super().__init__()

        conv = lambda : nn.Conv2d(
            in_channels = chan,
            out_channels = chan,
            kernel_size = kernel_size,
            padding = 'same',
            # in those values that padding creates, copy the values from the nearest boundary
            # https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html?highlight=replicate%20padding
            padding_mode = 'replicate'
        )

        self.net = nn.Sequential(
            conv(), activation(), conv(), activation()
        )

    def forward(self, x):
        
        return x + self.net(x) 

class Downsample(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_chan, 
            out_channels = out_chan,
            kernel_size = 2,
            stride = 2)

    def forward(self, x):
        return self.conv.forward(x)


class Upsample(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels = in_chan,
            out_channels = out_chan,
            kernel_size = 2,
            stride = 2,
            )


    def forward(self, x):
        return self.conv.forward(x)

class ChangeChannels(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size = 3):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels = in_chan,
            out_channels = out_chan,
            kernel_size = kernel_size,
            padding = 'same',
            )
    def forward(self, x):
        return self.conv.forward(x)
    
    
class CNNAutoencoder(pl.LightningModule):
    r""" 
    Parameters
    ----------
    dims: list of int
        Dimensions of input 
    layers: list of int

    """
    def __init__(self,
            dims = (1, 674, 434), 
            layers = [3, 3, 3, 3, 3],
            max_channels = 32,
            fc_layers = [10,10,10],
            kernel_size = 3,	
        ):
        super().__init__()
        self.dims = dims
        activation = lambda : torch.nn.LeakyReLU()
        self.save_hyperparameters()

        #######################################################################
        # Encoder	
        #######################################################################
        X = torch.zeros([1] + list(self.dims))
        
        ## CONVOLUTIONAL NEURAL NETWORK LAYERS##
        encoderCNN_layers = []
        ae_layer_dims = [nn.Sequential(*encoderCNN_layers).forward(X).shape]
        encoderCNN_dims = [nn.Sequential(*encoderCNN_layers).forward(X).shape]
        
        #Loop that constructs a concatenation of [(ResNetBlocks x lay) --> Downsample] x number of layers in "layers"
        for i, lay in enumerate(layers):
            in_chan = min(dims[0] * 2**i, max_channels)
            out_chan = min(dims[0] * 2**(i+1), max_channels)
            encoderCNN_layers += [ResNetBlock(in_chan) for _ in range(lay)]
            encoderCNN_layers += [Downsample(in_chan, out_chan)]
            ae_layer_dims += [nn.Sequential(*encoderCNN_layers).forward(X).shape]
            encoderCNN_dims += [nn.Sequential(*encoderCNN_layers).forward(X).shape]
            
        self.encoder_cnn = nn.Sequential(*encoderCNN_layers)
       
	#Shape before flattening
        cnn_last_layer_dims = torch.Size(ae_layer_dims[-1][1:])


	## FLATTEN LAYER ##
        X1 = self.encoder_cnn.forward(X)
	#Flatten the dimension of the last [ResNet--> Downsample] layer
        self.encoder_flatten = nn.Sequential(*[nn.Flatten(start_dim = 1)])
        ae_layer_dims += [nn.Sequential(*self.encoder_flatten).forward(X1).shape]
        
        # the dimensions of the flattened vector
        flat_dim = cnn_last_layer_dims[0]*cnn_last_layer_dims[1]*cnn_last_layer_dims[2]#n_channels x height x width of last layer

        ## LINEAR FULLY CONNECTED LAYERS ##
        X2 = self.encoder_flatten.forward(X1)
        encoderLFC_layers = []
        if len(fc_layers) > 0:
            encoderLFC_layers += [nn.Linear(flat_dim, fc_layers[0])]
            ae_layer_dims += [nn.Sequential(*encoderLFC_layers).forward(X2).shape]
        for l_in, l_out in zip(fc_layers[0:-1], fc_layers[1:]):
            encoderLFC_layers += [activation(), nn.Linear(l_in, l_out)]
            ae_layer_dims += [nn.Sequential(*encoderLFC_layers).forward(X2).shape]
            
        self.encoder_lfc = nn.Sequential(*encoderLFC_layers)
        
        Z = self.encoder_lfc.forward(X2)
        
        #######################################################################
        # Decoder	
        #######################################################################
        decoderLFC_layers = []
        
        ## LINEAR FULLY CONNECTED LAYERS ##
        for l_in, l_out in zip(fc_layers[-1:0:-1], fc_layers[-2::-1]):
            decoderLFC_layers += [nn.Linear(l_in, l_out), activation()]
            ae_layer_dims += [nn.Sequential(*decoderLFC_layers).forward(Z).shape]
        if len(fc_layers) > 0:
            decoderLFC_layers += [nn.Linear(fc_layers[0], flat_dim)]
            ae_layer_dims += [nn.Sequential(*decoderLFC_layers).forward(Z).shape]
        
        self.decoder_lfc = nn.Sequential(*decoderLFC_layers)
        
        ## UNFLATTEN LAYER ##   
        Y1 = self.decoder_lfc.forward(Z)
        self.decoder_unflatten = nn.Sequential(*[nn.Unflatten(1, cnn_last_layer_dims)])
        ae_layer_dims += [nn.Sequential(*self.decoder_unflatten).forward(Y1).shape]

        ## CONVOLUTIONAL NEURAL NETWORK LAYERS##
        Y2 = self.decoder_unflatten.forward(Y1)
        decoderCNN_layers = []
        for i, lay in reversed(list(enumerate(layers))):
            in_chan = min(dims[0] * 2**(i+1), max_channels)
            out_chan = min(dims[0] * 2**i, max_channels)
            decoderCNN_layers += [Upsample(in_chan, out_chan), nn.Upsample(size = encoderCNN_dims[i][2:]) ]
            decoderCNN_layers += [ResNetBlock(out_chan) for _ in range(lay)]
            ae_layer_dims += [nn.Sequential(*decoderCNN_layers).forward(Y2).shape]
        
        self.decoder_cnn = nn.Sequential(*decoderCNN_layers)

    def encode(self, x):

        #CNN step
        x = self.encoder_cnn.forward(x)
        #Flattening step
        x = self.encoder_flatten.forward(x)
        #Linear fully connected step
        z = self.encoder_lfc.forward(x)
    
        return z

    def decode(self, z):
    
        #Linear fully connected step
        z = self.decoder_lfc.forward(z)
        #Unflattening step
        z = self.decoder_unflatten.forward(z)       
        #CNN step
        x = self.decoder_cnn.forward(z) 
        
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    #### TRAINING ####
    def training_step(self, batch, batch_idx):
        X, time = batch
        X_hat = self.forward(X)
        loss = F.mse_loss(X_hat, X)
        self.log('train_loss', loss, on_step = True, on_epoch = True, logger = True)
        return loss
        
    def _shared_eval_step(self, X, batch_idx):
        X_hat = self.forward(X)
        loss = F.mse_loss(X_hat, X)

        return loss

    #### VALIDATION ####
    def validation_step(self, batch, batch_idx):
        X, time = batch
        loss = self._shared_eval_step(X, batch_idx)
        self.log('val_loss', loss, on_step = True, on_epoch = True, logger = True)
        return loss

    def validation_step_end(self, batch_parts):
        return torch.sum(batch_parts)

    def validation_epoch_end(self, outputs):
        self.log('val_loss_epoch', sum(outputs))

    #### TESTING ####
    def test_step(self, batch, batch_idx):
        X, time = batch
        loss = self._shared_eval_step(X, batch_idx)
        self.log('test_loss', loss, on_step = True, on_epoch = True, logger = True)
        return loss
     
    #### PREDICTION ####
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, time = batch
        z_hat = self.encode(X)
        X_hat = self.decode(z_hat)

        return X_hat, z_hat, time

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.5),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss_epoch",
                "strict": True,
            }
        }

if __name__ == '__main__':
    from torchinfo import summary
    model = CNNAutoencoder()
    input_shape = (1, 1, 674, 434)
    summary(model, input_shape, depth = 4, col_names = ['input_size', 'output_size', 'num_params', 'mult_adds'])    
