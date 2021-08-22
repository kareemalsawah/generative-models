import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *
from .resnet import *

class ConditionalInstanceNormal(nn.Module):
    def __init__(self,num_groups,num_features):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups

        self.weight = nn.Parameter(torch.ones((self.num_groups,self.num_features)))
        self.bias = nn.Parameter(torch.zeros((self.num_groups,self.num_features)))
        if self.num_features > 1:
            self.alpha = nn.Parameter(torch.zeros((self.num_groups,self.num_features)))
    
    def forward(self,x,condition):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, self.num_features, H, W)

        condition: torch.FloatTensor, shape = (batch_size) or shape = (batch_size,1)

        '''
        b_s = x.shape[0]
        condition = condition.reshape(-1).type(torch.LongTensor)
        assert torch.max(condition) < self.num_groups and torch.min(condition) > -1

        mean = torch.mean(x,dim=(2,3))
        diff = x-mean.reshape(b_s,self.num_features,1,1)
        diff_sq = diff*diff
        var = torch.mean(diff_sq,dim=(2,3)).reshape(b_s,self.num_features,1,1)

        normalized = diff/(torch.sqrt(var)+1e-5)

        weights = self.weight[condition].reshape(b_s,self.num_features,1,1)
        biases = self.bias[condition].reshape(b_s,self.num_features,1,1)

        if self.num_features > 1:
            m = torch.mean(mean,dim=1).reshape(b_s,1,1,1)
            v = torch.std(mean,dim=1).reshape(b_s,1,1,1)
            alphas = self.alpha[condition].reshape(b_s,self.num_features,1,1)

            shifted = normalized*weights + biases + alphas*(mean.reshape(b_s,self.num_features,1,1)-m)/(v+1e-5)
        else:
            shifted = normalized*weights + biases

        return shifted

class score_function_2d(nn.Module):
    def __init__(self,in_shape,num_channels=64,num_layers=16,L=10, use_cond_norm=False):
        super().__init__()
        self.in_shape = in_shape
        C,H,W = in_shape
        layers = [nn.Conv2d(C,num_channels,kernel_size=(3,3),padding=1),nn.BatchNorm2d(num_channels),nn.LeakyReLU()]
        for i in range(num_layers):
            layers.append(nn.Conv2d(num_channels,num_channels,kernel_size=(3,3),padding=1))
            if use_cond_norm:
                layers.append(ConditionalInstanceNormal(L,num_channels))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(num_channels,C,kernel_size=(3,3),padding=1))
    
        self.fc = nn.Sequential(*layers)
    
    def forward(self,x,sigmas):
        for layer in self.fc:
            if type(layer) is ConditionalInstanceNormal:
                x = layer.forward(x,sigmas)
            else:
                x = layer.forward(x)
        
        return x

class NCSN(nn.Module):
    def __init__(self, in_shape, L:int=10, sigma_1:float=10, n_bits:int=1):
        super().__init__()
        self.in_shape = in_shape
        self.n_bits = n_bits
        self.n_vals = 2**n_bits
        self.L = L
        self.sigma_1 = sigma_1

        D = in_shape[0]*in_shape[1]*in_shape[2]

        # Calculate geometric ratio for the sigmas
        ratio = 0.01

        # Add the different sigmas to the model
        sigmas = []
        for i in range(L):
            sigmas.append(sigma_1*(ratio**i))
        self.sigmas = np.array(sigmas)


        self.score_net = score_function_2d(in_shape, use_cond_norm=True)
        #self.score_net = SimpleResnet(in_shape[0],in_shape[0],n_filters=64,n_blocks=4)
    
    def forward(self, x, scale):
        C,H,W = self.in_shape
        x = x.reshape(-1,C,H,W)
        x = self.score_net(x, scale)

        return x
    
    def loss(self, x, num_samples=1):
        device = next(self.parameters()).device
        b_size = x.shape[0]
        x = x.reshape(b_size,-1)
        dim = x.shape[1]
        n_s = num_samples
        L, sigmas = self.L, self.sigmas

        sigmas = torch.tensor(sigmas).type(torch.FloatTensor).reshape(1,1,-1,1).to(device)
        noise = torch.normal(0,1,(b_size,n_s,L,dim)).to(device)

        x_bar = x.reshape(b_size,1,1,dim) + sigmas*noise
        sigmas_val = torch.arange(0,L).type(torch.FloatTensor).reshape(1,1,-1,1).to(device)
        sigmas_bs = torch.ones((b_size,n_s,L,1)).to(device)*sigmas_val
        sigmas_bs = sigmas_bs.reshape(-1,1)
        pred = self.forward(x_bar.reshape(b_size*n_s*L,dim),sigmas_bs).reshape(b_size,n_s,L,dim)
        loss = sigmas*pred + noise
        loss_sq = loss*loss
        loss = torch.sum(loss_sq.reshape(b_size,-1),dim=1)
        total_loss = torch.mean(loss)
        return 0.5*total_loss/L/num_samples

    def sample(self, num_gen:int, T:int=100, epsilon:float=2e-5):
        sigmas = self.sigmas
        start_dist = torch.distributions.Uniform(0,1)
        dim = self.in_shape[0]*self.in_shape[1]*self.in_shape[2]
        L = len(sigmas)
        C,H,W = self.in_shape
        device = next(self.parameters()).device
        
        x = start_dist.sample((num_gen,dim)).to(device)
        sigmas_sq = sigmas*sigmas
        with torch.no_grad():
            for l in range(L):
                sigma_i = sigmas[l]
                alpha = epsilon*sigma_i/sigmas[-1]
                #alpha = epsilon
                for t in range(T):
                    z_t = torch.normal(0,1,(num_gen,dim)).to(device)
                    sigs = torch.ones((num_gen)).to(device)*l
                    x = x + alpha/2*self.forward(x,sigs.reshape(-1,1)).reshape(num_gen,dim) + np.sqrt(alpha)*z_t
        
        return x.cpu().numpy().reshape(num_gen,C,H,W)
