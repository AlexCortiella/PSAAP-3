import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
import os

import ipdb

#### Functions ####
def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers
    
#### MODEL ####

class Encoder(nn.Module):
    def __init__(self,input_dim: int, inter_dims, latent_dim: int=4):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(input_dim,inter_dims[0]),
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )

        self.mu_l=nn.Linear(inter_dims[-1],latent_dim)
        self.log_sigma2_l=nn.Linear(inter_dims[-1],latent_dim)

    def forward(self, x):
        e=self.encoder(x)

        mu=self.mu_l(e)
        log_sigma2=self.log_sigma2_l(e)

        return mu,log_sigma2


class Decoder(nn.Module):
    def __init__(self,input_dim: int, inter_dims, latent_dim: int=4):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(latent_dim,inter_dims[-1]),
            *block(inter_dims[-1],inter_dims[-2]),
            *block(inter_dims[-2],inter_dims[-3]),
            nn.Linear(inter_dims[-3],input_dim)
        )



    def forward(self, z):
        x_rec_mean=self.decoder(z)

        return x_rec_mean

class VaDE(LightningModule):
    """Variational Deep Embedding  with Gaussian Prior and approx posterior.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 4,
                 nClusters: int = 2,
                 inter_dims = [32, 16, 8],
                 learning_rate = 1e-3,
                 cuda = False,
                 **kwargs):
                 
        super(VaDE,self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nClusters = nClusters
        self.inter_dims = torch.tensor(inter_dims)
        self.encoder=Encoder(self.input_dim, self.inter_dims, self.latent_dim)
        self.decoder=Decoder(self.input_dim, self.inter_dims, self.latent_dim)

        self.pi_vect=nn.Parameter(torch.FloatTensor(self.nClusters).fill_(1)/self.nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(self.nClusters,self.latent_dim).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(self.nClusters,self.latent_dim).fill_(0),requires_grad=True)
        
        
        self.lr = learning_rate
        self.cuda = cuda
        self.nClusters = nClusters
        self.input_dim = input_dim

    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = torch.exp(self.pi_vect) / torch.sum(torch.exp(self.pi_vect))
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def forward(self, x):
        #Mean and variance of latent
        z_mu, z_sigma2_log = self.encoder(x)
        #Sample latent
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        #Cluster probabilities
        pi = torch.exp(self.pi_vect) / torch.sum(torch.exp(self.pi_vect))
        #Mean and variance of cluster Gaussian
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c
        
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        return self.decoder(z)

    def _run_step(self, x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu
        x_rec_mean = self.decoder(z)

        return x_rec_mean
    
    def step(self, batch, batch_idx):
    
        det=1e-10#for numerical stability
        
        x, y = batch
        
        #Encode input data into latent space
        z_mu, z_sigma2_log = self.encoder(x)
        
        #Generate latent sample (reparametrization trick)
        z = torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu#Make sure there are as many samples as batch_size

        #Decode sampled latent variable into input
        x_rec_mean = self.decoder(z)

        loss_rec = -F.mse_loss(x, x_rec_mean)
        
        #Extract learnable parameters
        pi = torch.exp(self.pi_vect) / torch.sum(torch.exp(self.pi_vect))
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c
                
        #Compute gamma
        gamma=torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z,mu_c,log_sigma2_c)) + det
        gamma=gamma/(gamma.sum(1).view(-1,1))#batch_size x Clusters
        
        #Compute regularization terms in ELBO ###(2*pi constant terms dropped, same minimizer)
        loss_reg1 = 0.5*torch.mean(torch.sum(gamma*torch.sum(torch.log(2*torch.pi*torch.ones_like(log_sigma2_c.unsqueeze(0))) + log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        loss_reg2 = -0.5*torch.mean(torch.sum(1 + torch.log(2*torch.pi*torch.ones_like(z_sigma2_log)) + z_sigma2_log,1)) 

        loss_reg3 = -torch.mean(torch.sum(gamma*torch.log(pi.unsqueeze(0)/(gamma)),1))

        #Total loss (ELBO)
        loss_elbo = loss_rec + loss_reg1 + loss_reg2 + loss_reg3
        
        #for name, param in self.named_parameters():
            #if param.requires_grad:
                #print(name, param.data)
        
        return loss_elbo

    def training_step(self, batch, batch_idx):
        print(f'Gradient of pi_vect: {self.pi_vect.grad}')
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def latent_features(self, data_loader, return_labels=False):
        """Obtain latent features learnt by the model
        Args:
            data_loader: (DataLoader) loader containing the data
            return_labels: (boolean) whether to return true labels or not
        Returns:
           features: (array) array containing the features from the data
        """
        N = len(data_loader.dataset)
        features = np.zeros((N, self.latent_dim))
        if return_labels:
            true_labels = np.zeros(N, dtype=np.int64)
        start_ind = 0
        with torch.no_grad():
            for (data, labels) in data_loader:
                if self.cuda == 1:
                    data = data.cuda()
                # flatten data
                data = data.view(data.size(0), -1)  
                out = self.encoder(data)
                latent_feat = out['z']
                end_ind = min(start_ind + data.size(0), N+1)

                # return true labels
                if return_labels:
                    true_labels[start_ind:end_ind] = labels.cpu().numpy()
                features[start_ind:end_ind] = latent_feat.cpu().detach().numpy()  
                start_ind += data.size(0)
        if return_labels:
            return features, true_labels
        return features
    
    def plot_latent_space(self, data_loader, save=False):
        """Plot the latent space learnt by the model

        Args:
            data: (array) corresponding array containing the data
            labels: (array) corresponding array containing the labels
            save: (bool) whether to save the latent space plot

        Returns:
            fig: (figure) plot of the latent space
        """
        # obtain the latent features
        features, labels = self.latent_features(data_loader,return_labels=True)
        latent_dim = features.shape[1]
        list_latent_vars = [f'z{i+1}' for i in range(latent_dim)]
        allcomb = list(combinations_with_replacement(list_latent_vars, 2))
        plotlist = [item for item in allcomb if item[0] != item[1]]
        num_plots = len(plotlist)
        
        max_cols = 3
        quot = num_plots // max_cols
        rem = num_plots%max_cols
        if num_plots <= max_cols:
            cols = num_plots
            rows = 1
        elif rem == 0:
            cols = max_cols
            rows = quot
        else:
            cols = max_cols
            rows = quot + 1
        
        # plot only the first 2 dimensions
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize = (18,12))
        i = 0
        for ax in axes.flatten():
            if i < num_plots:
                xlab, ylab = plotlist[i]
                idx1, idx2 = int(xlab[-1])-1, int(ylab[-1])-1
                ax.scatter(features[:, idx1], features[:, idx2], c=labels, marker='o',
                    edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
                ax.set_xlabel(xlab,fontsize = 20)
                ax.set_ylabel(ylab,fontsize = 20)
            #plt.colorbar()
            i += 1
        fig.tight_layout()
#         if(save):
#             fig.savefig('latent_space.png')

    def reconstruct_data(self, data_loader, sample_size):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        # sample random data from loader
        indices = np.random.randint(0, len(data_loader.dataset), size=sample_size)
        test_random_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=sample_size, sampler=SubsetRandomSampler(indices))
  
        # obtain values
        it = iter(test_random_loader)
        test_batch_data, _ = it.next()
        original = test_batch_data.data.numpy()
        if self.cuda:
            test_batch_data = test_batch_data.cuda()  

        # obtain reconstructed data  
        out = self.forward(test_batch_data) 
        
        reconstructed = out['x_rec']
        
        return original, reconstructed.data.cpu().numpy()
    
    def plot_reconstruction(self, data_loader, sample_size = -1):
        
        original, reconstructed = self.reconstruct_data(data_loader, sample_size)
            
        num_plots = sample_size    
        max_cols = 3
        quot = num_plots // max_cols
        rem = num_plots%max_cols
        if num_plots <= max_cols:
            cols = num_plots
            rows = 1
        elif rem == 0:
            cols = max_cols
            rows = quot
        else:
            cols = max_cols
            rows = quot + 1
        
        # plot only the first 2 dimensions
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize = (18,12))
        i = 0
        for ax in axes.flatten():
            if i < num_plots:
                ax.plot(original[i,:],'r')
                ax.plot(reconstructed[i,:],'b')
                ax.set_xlabel('Time',fontsize = 20)
                ax.set_ylabel('x',fontsize = 20)
                ax.legend(['Original', 'Reconstructed'],fontsize = 20)
            #plt.colorbar()
            i += 1
        fig.tight_layout()
        
    def generate_sample(self, num_samples):
        
        p = torch.distributions.Normal(torch.zeros((self.latent_dim)), torch.ones((self.latent_dim)))
        X = []
        Z = []
        
        for s in range(num_samples):
            z = p.sample()
            Z.append(z)
            x = self.decoder(z)
            X.append(x)
        return X, Z
