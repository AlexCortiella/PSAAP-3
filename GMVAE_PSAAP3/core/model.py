import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


#### MODEL ####

class VEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, layers):
        super(VEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, layers[0])
        self.linear2 = nn.Linear(layers[0], layers[1])
        self.linear22 = nn.Linear(layers[1], layers[2])
        self.linear3 = nn.Linear(layers[2], latent_dim)
        self.linear4 = nn.Linear(layers[2], latent_dim)
        
    def sample(self, mu, sigma):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        q = torch.distributions.Normal(mu, sigma)
        z = q.rsample()
        return p, q, z
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear22(x))
        mu =  self.linear3(x)
        sigma = torch.exp(self.linear4(x))
        p, q, z = self.sample(mu, sigma)
        
        output = {'z_mean': mu, 'z_sigma': sigma, 'z': z, 'z_prior': p, 'z_posterior': q}
        
        return output
    
class VDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, layers):
        super(VDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, layers[2])
        self.linear2 = nn.Linear(layers[2], layers[1])
        self.linear22 = nn.Linear(layers[1], layers[0])
        self.linear3 = nn.Linear(layers[0], input_dim)
        
    def forward(self, z):
        z = F.elu(self.linear1(z))
        z = F.elu(self.linear2(z))
        z = F.elu(self.linear22(z))
        z = F.sigmoid(self.linear3(z))
        
        output = {'x_rec': z}
        
        return output

class VAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        layers = torch.tensor([32, 16, 8]),
        kl_coeff: float = 0.1,
        learning_rate: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_dim: input dimension
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()
        
        self.save_hyperparameters()

        self.lr = learning_rate
        self.kl_coeff = kl_coeff
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.layers = layers
        self.cuda = False

        self.encoder = VEncoder(input_dim,latent_dim, layers)
        self.decoder = VDecoder(input_dim,latent_dim, layers)
        


    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        out_inf = self.encoder(x)
        z = out_inf['z']
        return self.decoder(z)

    def _run_step(self, x):
        out_inf = self.encoder(x)
        z = out_inf['z']
        out_gen = self.decoder(z)
        
        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
    
    def step(self, batch, batch_idx):
        
        # For cyclic KL weight
        if self.current_epoch%750<500:
            self.kl_coeff = 1.0/500.0*float((self.current_epoch)%750) 
        else: 
            self.kl_coeff = 1.0
            
        x, y = batch
        out = self._run_step(x)
        
        z = out['z']
        x_hat = out['x_rec']
        p = out['z_prior']
        q = out['z_posterior']

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        self.log("Total loss", logs['loss'], on_epoch=True)
        self.log("Reconstruction loss", logs['recon_loss'], on_epoch=True)
        self.log("wKL", logs['kl'], on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
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
        
        