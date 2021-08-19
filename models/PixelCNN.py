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


class ResBlockMasked(nn.Module):
    def __init__(self,type_A, num_channels, kernel_size):
        super().__init__()
        h = num_channels//2

        if h < 1: h = 1

        self.conv_1 = nn.Conv2d(num_channels, h, kernel_size=(1,1))
        self.conv_2 = MaskedConv2d(type_A, h, h, kernel_size=kernel_size, padding=(kernel_size[0]-1)//2)
        self.conv_3 = nn.Conv2d(h,num_channels, kernel_size=(1,1))

        self.non_linearity = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(h)
        self.bn2 = nn.BatchNorm2d(h)
        self.bn3 = nn.BatchNorm2d(num_channels)
    
    def forward(self,x):
        out = self.non_linearity(self.bn1(self.conv_1(x)))
        out = self.non_linearity(self.bn2(self.conv_2(out)))
        out = self.non_linearity(self.bn3(self.conv_3(out)))

        out += x  # Residual connection

        return out

class PixelCNN(nn.Module):
    def __init__(self, in_shape, n_blocks=8, n_bits=1):
        super().__init__()
        C, H, W = in_shape
        self.H = H
        self.W = W
        self.C = C
        self.n_bits = n_bits
        self.n_vals = 2**n_bits

        self.conv_1 = nn.Sequential(MaskedConv2d(True, C, 120, (7, 7), padding=3),
                                    nn.BatchNorm2d(120),
                                    nn.ReLU())

        res_layers = []
        for i in range(n_blocks):
            res_layers.append(ResBlockMasked(False, 120, (7, 7)))

        self.res_layers = nn.Sequential(*res_layers)

        self.out_layer = nn.Sequential(MaskedConv2d(False, 120, 60, (1, 1)),
                                       nn.BatchNorm2d(60),
                                       nn.ReLU(),
                                       MaskedConv2d(False, 60, C*self.n_vals, (1, 1)))
        self.log_softmax = nn.LogSoftmax(dim=4)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size,self.C,self.H,self.W)
        '''
        batch_size = x.shape[0]
        out = self.conv_1(x)

        out = self.res_layers.forward(out)

        out = self.out_layer.forward(out).reshape(batch_size, self.C, self.n_vals, self.H, self.W)
        out = out.permute(0,1,3,4,2)  # shape = (batch_size,C,H,W,4)

        out = self.log_softmax(out)

        return out
    
    def loss(self, target):
        device = target.device

        pred = self.forward(target).reshape(-1, self.n_vals)
        target = target.type(torch.LongTensor).to(device).reshape(-1)
        
        return F.nll_loss(pred, target)