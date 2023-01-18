#MODEL

###### MODEL CLASS ######

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from  torch.distributions.categorical import Categorical

import pandas as pd
import numpy as np

#Autoencoder definition
class Encoder(nn.Module):

    def __init__(self, encoded_space_dim=2, fc2_input_dim=128, num_channels=[8, 16, 32]):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        self.num_channels = num_channels

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, self.num_channels[0], 9, stride=6, padding=0),
            nn.BatchNorm2d(self.num_channels[0]),
            nn.ReLU(),
            nn.Conv2d(self.num_channels[0], self.num_channels[1], 9, stride=6, padding=0),
            nn.BatchNorm2d(self.num_channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.num_channels[1], self.num_channels[2], 3, stride=2, padding=0),
            nn.ReLU()
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(8 * 5 * self.num_channels[2], fc2_input_dim),
            nn.ReLU(),
            nn.Linear(fc2_input_dim, encoded_space_dim * 2)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        mu, logsigmasq = x[:, :self.encoded_space_dim], x[:, self.encoded_space_dim:]
        return mu, logsigmasq


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim=2, fc2_input_dim=128, num_channels=[32, 16, 8]):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(),
            nn.Linear(fc2_input_dim, 8 * 5 * num_channels[0]),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(num_channels[0], 8, 5))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(num_channels[0], num_channels[1], 3,
                               stride=2, output_padding=(1, 0)),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[1], num_channels[2], 9, stride=6,
                               padding=0, output_padding=(0, 2)),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[2], 2, 9, stride=6,
                               padding=0, output_padding=(5, 5))
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        mu, logsigmasq = x[:, 0, :, :], x[:, 1, :, :]
        return mu, logsigmasq

#GMVAE intermediate functions

class GMVAE(nn.Module):
    
    
    def __init__(self, cfg):
        super().__init__()
        # initialize latent GMM model parameters
        
        self.latent_dim = cfg.latent_dim
        self.n_clusters = cfg.n_clusters
        self.n_modalities = cfg.n_modalities
        self.num_epochs = cfg.total_epochs
        self.params = {}
        self.pi_variables = torch.nn.Parameter(torch.zeros(self.n_clusters), requires_grad = True)
        self.params['pi_c'] = torch.exp(self.pi_variables) / torch.sum(torch.exp(self.pi_variables))
        self.params['mu_c'] = torch.rand((self.n_clusters, self.latent_dim))
        self.params['logsigmasq_c'] = torch.randn((self.n_clusters, self.latent_dim))
        
        # initialize neural networks
        self.encoder_list = []
        self.decoder_list = []
        self.trainable_parameters = []
        self.trainable_parameters.append(self.pi_variables)

        for _ in range(self.n_modalities):
            encoder = Encoder(encoded_space_dim=self.latent_dim)
            decoder = Decoder(encoded_space_dim=self.latent_dim)
            self.encoder_list.append(encoder)
            self.decoder_list.append(decoder)
            self.trainable_parameters += list(encoder.parameters()) + list(decoder.parameters())

        self.trainable_parameters = nn.ParameterList(self.trainable_parameters)
        # Utils
        self.em_reg = 1e-6
        
    def _encoder_step(self, x_list, encoder_list, decoder_list):
        """
        Maps D-modality data to distributions of latent embeddings.
        :param x_list: length-D list of (N, data_dim) torch.tensor
        :param encoder_list: length-D list of Encoder
        :param decoder_list: length-D list of Decoder
        :param params: dictionary of non-DNN parameters
        :return:
            mu: (N, latent_dim) torch.tensor containing the mean of embeddings
            sigma: (N, latent_dim) torch.tensor containing the std dev of embeddings
        """

        assert(len(encoder_list) == len(decoder_list))
        # assert (len(encoder_list) == len(x_list))

        if len(encoder_list) == 1:
            mu, logsigmasq = encoder_list[0].forward(x_list[0])

        else:
            # compute distribution of qz as product of experts
            qz_inv_var = 0
            qz_mean_inv_var = 0

            for d, encoder in enumerate(encoder_list):
                mu_, logsigmasq_ = encoder.forward(x_list[d])
                qz_inv_var += torch.exp(-logsigmasq_)
                qz_mean_inv_var += mu_ * torch.exp(-logsigmasq_)

            mu = qz_mean_inv_var / qz_inv_var  # mu = qz_mean
            logsigmasq = - torch.log(qz_inv_var)  # sigma = qz_stddev

        return mu, logsigmasq
    
    def _em_step(self, z, mu, update_by_batch=False):
        # compute gamma_c ~ p(c|z) for each x
        mu_c = self.params['mu_c']  # (K, Z)
        logsigmasq_c =self.params['logsigmasq_c']  # (K, Z)
        sigma_c = torch.exp(0.5 * logsigmasq_c)
        pi_c = self.params['pi_c']

        log_prob_zc = Normal(mu_c, sigma_c).log_prob(z.unsqueeze(dim=1)).sum(dim=2) + torch.log(pi_c)  #[N, K]
        log_prob_zc -= log_prob_zc.logsumexp(dim=1, keepdims=True)
        gamma_c = torch.exp(log_prob_zc) + self.em_reg
        
        denominator = torch.sum(gamma_c, dim=0).unsqueeze(1)
        mu_c = torch.einsum('nc,nz->cz', gamma_c, mu) / denominator
        logsigmasq_c = torch.log(torch.einsum('nc,ncz->cz', gamma_c, (mu.unsqueeze(dim=1) - mu_c) ** 2)) - torch.log(denominator)
        
        if not update_by_batch:
            return gamma_c, mu_c, logsigmasq_c

        else:
            hist_weights = self.params['hist_weights']
            hist_mu_c = self.params['hist_mu_c']
            hist_logsigmasq_c = self.params['hist_logsigmasq_c']

            curr_weights = denominator
            new_weights = hist_weights + curr_weights
            new_mu_c = (hist_weights * hist_mu_c + curr_weights * mu_c) / new_weights
            new_logsigmasq_c = torch.log(torch.exp(torch.log(hist_weights) + hist_logsigmasq_c) +
                                         torch.exp(torch.log(curr_weights) + logsigmasq_c)) - torch.log(new_weights)

            self.params['hist_weights'] = new_weights
            self.params['hist_mu_c'] = new_mu_c
            self.params['hist_logsigmasq_c'] = new_logsigmasq_c
            return gamma_c, new_mu_c, new_logsigmasq_c
        
        
    def _decoder_step(self, x_list, z, encoder_list, decoder_list, mu, logsigmasq, gamma_c):
        """
        Computes a stochastic estimate of the ELBO.
        :param x_list: length-D list of (N, data_dim) torch.tensor
        :param z: MC samples of the encoded distributions
        :param encoder_list: length-D list of Encoder
        :param decoder_list: length-D list of Decoder
        :param params: dictionary of non-DNN parameters
        :return:
            elbo: (,) tensor containing the elbo estimation
        """
        assert(len(encoder_list) == len(decoder_list))

        sigma = torch.exp(0.5 * logsigmasq)
        mu_c = self.params['mu_c']
        logsigmasq_c = self.params['logsigmasq_c']
        pi_c = self.params['pi_c']
        
        elbo = 0
        for d, decoder in enumerate(decoder_list):
            mu_, logsigmasq_ = decoder.forward(z)
            elbo += Normal(mu_, torch.exp(0.5 * logsigmasq_)).log_prob(x_list[d]).sum()
        elbo += - 0.5 * torch.sum(gamma_c * (logsigmasq_c + (sigma.unsqueeze(1) ** 2 + (mu.unsqueeze(1) - mu_c) ** 2) /
                                             torch.exp(logsigmasq_c)).sum(dim=2))
        elbo += torch.sum(gamma_c * (torch.log(pi_c) - torch.log(gamma_c))) + 0.5 * torch.sum(1 + logsigmasq)
        
        N = x_list[0].shape[0]
        
        return 1/N * elbo
        
    def loss(self, batch_x):
        
        #Extract data modalities
        x_list = [batch_x]  # assume D=2 and each modality has data_dim
            
        #Assign pi_c
        pi_c = torch.exp(self.pi_variables) / torch.sum(torch.exp(self.pi_variables))
        self.params['pi_c'] = pi_c
        
        #Encode input data to Gaussian latent space with mu and logsigmaq
        mu, logsigmasq = self._encoder_step(x_list, self.encoder_list, self.decoder_list)

        #Sample from the latent space
        sigma = torch.exp(0.5 * logsigmasq)
        eps = Normal(0, 1).sample(mu.shape)
        z = mu + eps * sigma

        #Perform EM step to estimate mu_c and logsigmasq_c
        with torch.no_grad():
            gamma_c, mu_c, logsigmasq_c = self._em_step(z, mu, update_by_batch=True)
        self.params['mu_c'] = mu_c
        self.params['logsigmasq_c'] = logsigmasq_c

        #Compute elbo and loss
        elbo = self._decoder_step(x_list, z, self.encoder_list, self.decoder_list, mu, logsigmasq, gamma_c)
        loss = - elbo
        
        return loss
    
    def predict(self, batch_x):
        
        with torch.no_grad():
            #Extract data modalities
            x_list = [batch_x]  # assume D=2 and each modality has data_dim

            #Assign pi_c
            pi_c = torch.exp(self.pi_variables) / torch.sum(torch.exp(self.pi_variables))
            self.params['pi_c'] = pi_c

            #Encode input data to Gaussian latent space with mu and logsigmaq
            mu, logsigmasq = self._encoder_step(x_list, self.encoder_list, self.decoder_list)

            #Sample from the latent space
            sigma = torch.exp(0.5 * logsigmasq)
            eps = Normal(0, 1).sample(mu.shape)
            z = mu + eps * sigma

            # compute gamma_c ~ p(c|z) for each x
            mu_c = self.params['mu_c']  # (K, Z)
            logsigmasq_c =self.params['logsigmasq_c']  # (K, Z)
            sigma_c = torch.exp(0.5 * logsigmasq_c)
            pi_c = self.params['pi_c']

            log_prob_zc = Normal(mu_c, sigma_c).log_prob(z.unsqueeze(dim=1)).sum(dim=2) + torch.log(pi_c)  #[N, K]
            log_prob_zc -= log_prob_zc.logsumexp(dim=1, keepdims=True)
            gamma_c = torch.exp(log_prob_zc) + self.em_reg
            
            for d, decoder in enumerate(self.decoder_list):
                mu_, logsigmasq_ = decoder.forward(z)
                
            #Sample from the input space
            sigma_ = torch.exp(0.5 * logsigmasq_)
            eps_ = Normal(0, 1).sample(mu_.shape)
            x_rec = mu_ + eps_ * sigma_
            
        return mu, logsigmasq, z, gamma_c, mu_, logsigmasq_, x_rec
