#Alternative CNN AE model
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as spn
import pytorch_lightning as pl

'''
QuadConvLayer block

Input:
    point_dim: space dimension
    channels_in: input feature channels
    channels_out: output feature channels
    N_in: number of input points
    N_out: number of output points
    adjoint: downsample or upsample
    use_bias: add bias term to output of layer
    activation1:
    activation2:
'''
class ConvBlock(nn.Module):
    def __init__(self,
                    channels_in,
                    channels_out,
                    N_in = None,
                    N_out = None,
                    kernel_size = 2,
                    adjoint = False,
                    use_bias = False,
                    output_padding = 0,
                    activation1 = nn.CELU(alpha=1),
                    activation2 = nn.CELU(alpha=1)
                    ):
        super().__init__()

        self.adjoint = adjoint
        self.activation1 = activation1
        self.activation2 = activation2

        
        Conv1 = nn.Conv2d

        if self.adjoint:
            Conv2 = nn.ConvTranspose2d
        else:
            Conv2 = nn.Conv2d

        Norm = nn.InstanceNorm2d

        if self.adjoint:
            conv1_channel_num = channels_out
            #stride = tuple([int(np.floor((eout-1-(kernel_size-1))/(ein-1))) for eout, ein in zip(N_out, N_in)])
            stride = 2
            self.conv2 = Conv2(channels_in,
                                channels_out,
                                kernel_size,
                                stride=stride,
                                output_padding=output_padding
                                )
        else:
            conv1_channel_num = channels_in
            #stride = tuple([int(np.floor((ein-1-(kernel_size-1))/(eout-1))) for eout, ein in zip(N_out, N_in)])
            stride = 2
            self.conv2 = Conv2(channels_in,
                                channels_out,
                                kernel_size,
                                stride=stride
                                )

        self.conv1 = Conv1(conv1_channel_num,
                            conv1_channel_num,
                            kernel_size,
                            padding='same'
                            )

        self.batchnorm1 = Norm(conv1_channel_num)

        self.batchnorm2 = Norm(channels_out)

    '''
    Forward mode
    '''
    def forward_op(self, data):
        x = data

        x1 = self.conv1(x)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x

        x2 = self.conv2(x1)
        x2 = self.activation2(self.batchnorm2(x2))

        return x2

    '''
    Adjoint mode
    '''
    def adjoint_op(self, data):
        x = data

        x2 = self.conv2(x)
        x2 = self.activation2(self.batchnorm2(x2))

        x1 = self.conv1(x2)
        x1 = self.activation1(self.batchnorm1(x1))
        x1 = x1 + x2

        return x1

    '''
    Apply operator
    '''
    def forward(self, data):
        if self.adjoint:
            output = self.adjoint_op(data)
        else:
            output = self.forward_op(data)

        return output
'''
Encoder module
'''
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #establish block type
        Block = ConvBlock

        #specific args
        latent_dim = kwargs.pop('latent_dim')
        num_cnn_layers = kwargs.pop('num_cnn_layers')
        channel_seq = kwargs.pop('channel_seq')
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        #build network
        self.cnn = nn.Sequential()
        X = torch.zeros(input_shape)
        cnn_dims = []
        N_in = input_shape[2:]
        for i in range(num_cnn_layers):
            self.cnn.append(Block(channel_seq[i],
                                    channel_seq[i+1],
                                    N_in = N_in,
                                    N_out = tuple([int(np.floor(e/2))  for e in N_in]),
                                    activation1 = forward_activation(),
                                    activation2 = forward_activation(),
                                    **kwargs
                                    ))
            N_in = tuple([int(np.floor(e/2))  for e in N_in])
            cnn_dims.append(self.cnn(X).shape)
            
        self.cnn_out_shape = self.cnn(torch.zeros(input_shape)).shape

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(self.cnn_out_shape.numel(), latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation2)

        self.linear(self.flat(torch.zeros(self.cnn_out_shape)))

    def forward(self, x):
        
        x1 = torch.clone(x)
        for i, layer in enumerate(self.cnn):
            x1 = layer.forward(x1)
            print(f'CNN layer {i}: {x1.shape}')
            
        x = self.cnn(x)
        print(f'CNN output: {x.shape}')
        x = self.flat(x)
        print(f'Flatten output: {x.shape}')
        output = self.linear(x)
        print(f'Latent: {output.shape}')

        return output

'''
Decoder module
'''
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #establish block type
        Block = ConvBlock
        #specific args
        latent_dim = kwargs.pop('latent_dim')
        num_cnn_layers = kwargs.pop('num_cnn_layers')
        channel_seq = kwargs.pop('channel_seq')[::-1]
        input_shape = kwargs.pop('input_shape')

        #activations
        forward_activation = kwargs.pop('forward_activation')
        latent_activation = kwargs.pop('latent_activation')
        self.activation1 = latent_activation()
        self.activation2 = latent_activation()

        #build network
        self.unflat = nn.Unflatten(1, input_shape[1:])

        self.linear = nn.Sequential()

        self.linear.append(spn(nn.Linear(latent_dim, latent_dim)))
        self.linear.append(self.activation1)
        self.linear.append(spn(nn.Linear(latent_dim, input_shape.numel())))
        self.linear.append(self.activation2)

        self.cnn = nn.Sequential()
       
        N_in = input_shape[2:]
        for i in range(num_cnn_layers):
            output_padding = 2 if i == num_cnn_layers-1 else 0
            self.cnn.append(Block(channel_seq[i],
                                    channel_seq[i+1],
                                    N_in = N_in,
                                    N_out = 2 * N_in,
                                    activation1 = forward_activation() if i!=num_cnn_layers-1 else nn.Identity(),
                                    activation2 = forward_activation(),
                                    adjoint = True,
                                    output_padding = output_padding,
                                    **kwargs
                                    ))

    def forward(self, x):
        print(f'Latent: {x.shape}')
        x = self.linear(x)
        print(f'Linear output: {x.shape}')
        x = self.unflat(x)
        print(f'Unflatten output: {x.shape}')
        
        x1 = torch.clone(x)
        for i, layer in enumerate(self.cnn):
            x1 = layer.forward(x1)
            print(f'CNN layer {i}: {x1.shape}')
            
        output = self.cnn(x)
        
        print(f'CNN output: {output.shape}')
        return output
        
        
        
class CNNAutoencoder(pl.LightningModule):
    def __init__(self,
                    latent_dim = 10,
                    num_cnn_layers = 4,
                    channel_seq = [1, 4, 8, 16, 32],
                    forward_activation = nn.CELU,
                    latent_activation = nn.CELU,
                    output_activation = nn.Tanh,
                    loss_fn = nn.functional.mse_loss,
                    noise_scale = 0.0,
                    input_shape = (1, 1, 674, 434),
                    learning_rate = 1e-2,
                    profiler = None,
                    **kwargs
                    ):
                    
        super().__init__()

        #save model hyperparameters under self.hparams
        self.save_hyperparameters(ignore=['loss_fn',
                                            'noise_scale',
                                            'forward_activation',
                                            'latent_activation',
                                            'output_activation',
                                            'input_shape',
                                            'learning_rate',
                                            'profiler'])
                                            
        #model pieces
        self.encoder = Encoder(**self.hparams,
                                forward_activation=forward_activation,
                                latent_activation=latent_activation,
                                input_shape=input_shape)
                                
        self.decoder = Decoder(**self.hparams,
                                forward_activation=forward_activation,
                                latent_activation=latent_activation,
                                input_shape=self.encoder.cnn_out_shape)

        #training hyperparameters
        self.loss_fn = loss_fn
        self.noise_scale = noise_scale
        self.output_activation = output_activation()
        self.learning_rate = learning_rate

        self.profiler = profiler



    def encode(self, x):

        z = self.encoder.forward(x)
            
        return z

    def decode(self, z):
    
        x = self.decoder.forward(z) 

        return x

    def forward(self, x):
        print('Encoding...')
        z = self.encode(x)
        
        print('Decoding...')
        return self.decode(z)

    #### TRAINING ####
    def training_step(self, X, batch_idx):
        X_hat = self.forward(X)
        print(f'Reconstructed dimension: {X_hat.shape}')
        loss = self.loss_fn(X_hat, X)
        self.log('train_loss', loss, on_step = True, on_epoch = True, logger = True)
        return loss
        
    def _shared_eval_step(self, X, batch_idx):
        X_hat = self.forward(X)
        loss = self.loss_fn(X_hat, X)

        return loss

    #### VALIDATION ####
    def validation_step(self, X, batch_idx):
        loss = self._shared_eval_step(X, batch_idx)
        self.log('val_loss', loss, on_step = True, on_epoch = True, logger = True)
        return loss

    def validation_step_end(self, batch_parts):
        return torch.sum(batch_parts)

    def validation_epoch_end(self, outputs):
        self.log('val_loss_epoch', sum(outputs))

    #### TESTING ####
    def test_step(self, X, batch_idx):
        loss = self._shared_eval_step(X, batch_idx)
        self.log('test_loss', loss, on_step = True, on_epoch = True, logger = True)
        return loss
     
    #### PREDICTION ####
    def predict_step(self, X, batch_idx, dataloader_idx=0):
        z_hat = self.encode(X)
        X_hat = self.decode(z_hat)

        out_dict = {"reconstructed": X_hat, "latent": z_hat}

        return out_dict

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
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
