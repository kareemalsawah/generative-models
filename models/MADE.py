import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *

class LinearMask(nn.Module):
    '''
    
    '''
    def __init__(self,in_features,out_features,type_A=True,bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_A = type_A
        self.is_bias = bias
        k = np.sqrt(1/self.in_features)

        self.weight = nn.Parameter(uniform_dist(-k,k,(self.out_features,self.in_features)))
        if self.is_bias:
            self.bias = nn.Parameter(uniform_dist(-k,k,(self.out_features)))

        ones = torch.ones((self.out_features,self.in_features)).type(torch.FloatTensor)
        if self.type_A:
            self.mask = torch.tril(ones, diagonal=0)
        else:
            self.mask = torch.tril(ones, diagonal=-1)

        self.mask = nn.Parameter(self.mask)
        self.mask.requires_grad = False

    def forward(self,x):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, self.in_features)

        Returns
        -------
        out: torch.FloatTensor, shape = (batch_size, self.out_features)
        '''
        weights = self.weight*self.mask
        out = torch.matmul(x,weights.T) + self.bias.reshape(1,-1)
        return out
        
class MADE(nn.Module):
    '''
    MADE model to model H*W variables using a masked autoregressive model. Variables will be considered binary.
    This will be used to generate H*W sized images assuming raster scan ordering for the variables

    This model currently works only with binary images (1 binary value for each pixel = 1 channel)

    Attributes
    ----------
    H: int
        Height of the image in pixels
    W: int
        Width of the image in pixels
    '''
    def __init__(self, in_shape, n_bits=1):
        super().__init__()
        C,H,W = in_shape
        self.H = H
        self.W = W
        self.C = C
        self.n_bits = n_bits
        self.n_vals = 2**n_bits
        n_features = self.H*self.W

        self.layers = nn.Sequential(LinearMask(n_features,n_features,False),
                                    nn.LeakyReLU(),
                                    LinearMask(n_features,n_features,True),
                                    nn.LeakyReLU(),
                                    LinearMask(n_features,n_features,True),
                                    nn.LeakyReLU(),
                                    LinearMask(n_features,n_features,True),
                                    nn.Sigmoid())
    
    def forward(self, x):
        '''
        Forward pass through the masked autoregressive model

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, 1, self.H, self.W)
        
        Returns
        -------
        probs: torch.FloatTensor, shape = (batch_size, self.H*self.W)
            Each value represents P(pixel=1)
        '''
        batch_size = x.shape[0]
        return self.layers.forward(x.reshape(batch_size,-1)).reshape(x.shape)
    
    def log_prob(self,x):
        '''
        Estimate the nats per dim of given samples

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, 1, self.H, self.W)
        
        Returns
        -------
        torch.FloatTensor, shape = (batch_size)
            Each element is the nats per dim of the respective sample from x
        '''
        _,_,H,W = x.shape
        prob = self.forward(x)
        nats_per_dim = torch.sum(safe_log(prob),dim=(1,2,3))/(H*W)

        return nats_per_dim
    
    def loss(self, x):
        pred = self.forward(x).reshape(x.shape[0],-1)
        device = x.device
        x = torch.round(x.reshape(x.shape[0],-1)).type(torch.LongTensor).to(device)
        inv_pred = 1-pred
        inv_target = 1-x

        loss_0 = -inv_target*safe_log(inv_pred)
        loss_1 = -x*safe_log(pred)

        nats_per_dim = torch.sum(loss_0 + loss_1) / (pred.shape[0]*pred.shape[1])
        
        return nats_per_dim
