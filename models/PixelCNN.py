import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *

class MaskedConv2d(nn.Module):
    def __init__(self, type_A, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.type_A = type_A
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.is_bias = bias

        assert len(kernel_size) == 2
        assert kernel_size[0] == kernel_size[1]

        k = 1/(in_channels*kernel_size[0]*kernel_size[1])

        self.weights = nn.Parameter(uniform_dist(-k,k,(out_channels,in_channels,*kernel_size)))

        if self.is_bias:
            self.bias = nn.Parameter(uniform_dist(-k,k,(out_channels)))

        mask = torch.ones(kernel_size).type(torch.FloatTensor)
        idx = kernel_size[0]//2 + 1
        mask[idx:] = 0
        if self.type_A:  # No horizontal propagation
            mask[idx-1,idx-1:] = 0
        else:  # Allows horizontal propagation (type B)
            mask[idx-1,idx:] = 0
        
        self.mask = nn.Parameter(mask.reshape(1,1,*kernel_size))
        self.mask.requires_grad = False  # Freeze mask


    def forward(self,x):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, )
        '''
        masked_weights = self.weights*self.mask

        return F.conv2d(x, masked_weights, self.bias, self.stride,
                        self.padding, self.dilation, 1)


class PixelCNN(nn.Module):
    def __init__(self,H,W):
        super().__init__()
        self.H = H
        self.W = W

        self.layers = nn.Sequential(MaskedConv2d(True, 1, 64, (7,7), padding=3),
                                    nn.LeakyReLU(),
                                    MaskedConv2d(False, 64, 64, (7,7), padding=3),
                                    nn.LeakyReLU(),
                                    MaskedConv2d(False, 64, 64, (7,7), padding=3),
                                    nn.LeakyReLU(),
                                    MaskedConv2d(False, 64, 64, (7,7), padding=3),
                                    nn.LeakyReLU(),
                                    MaskedConv2d(False, 64, 64, (7,7), padding=3),
                                    nn.LeakyReLU(),
                                    MaskedConv2d(False, 64, 64, (7,7), padding=3),
                                    nn.LeakyReLU(),
                                    MaskedConv2d(False, 64, 64, (1,1)),
                                    nn.LeakyReLU(),
                                    MaskedConv2d(False, 64, 1, (1,1)),
                                    nn.Sigmoid()
                                    )
    
    def forward(self,x):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, self.H, self.W)

        Returns
        -------
        torch.FloatTensor, shape = (batch_size, self.H*self.W)
        '''
        batch_size = x.shape[0]
        out = self.layers.forward(x.reshape(batch_size,1,self.H,self.W))

        return out
    
    def log_prob(self, x):
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
        _,C,H,W = x.shape
        prob = self.forward(x)
        nats_per_dim = torch.sum(safe_log(prob),dim=(1,2,3))/(C*H*W)

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