import torch
import torch.nn as nn
import numpy as np

from .utils import *

class WeightedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))

    def forward(self, x):
        x = self.conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,n_filters):
        super().__init__()

        self.convs = nn.Sequential(nn.Conv2d(n_filters,n_filters,kernel_size=(1,1)),
                                    nn.BatchNorm2d(n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(n_filters,n_filters,kernel_size=(3,3),padding=1),
                                    nn.BatchNorm2d(n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(n_filters,n_filters,kernel_size=(1,1)))

    def forward(self,x):
        h = self.convs(x)
        return h + x

class SimpleResnet(nn.Module):
    def __init__(self,in_channels,out_channels,n_filters=128,n_blocks=8,identity_init=False):
        super().__init__()

        self.conv_1 = WeightedConv2d(in_channels,n_filters,kernel_size=(3,3),padding=1)

        layers = []
        for _ in range(n_blocks):
            layers.append(ResBlock(n_filters))
        layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)

        self.conv_2 = WeightedConv2d(n_filters,out_channels,kernel_size=(3,3),padding=1)
        if identity_init:
            self.conv_2.weight.data *= 0
            self.conv_2.bias.data *= 0

    def forward(self,x):
        x = self.conv_1(x)
        x = self.layers(x)
        x = self.conv_2(x)

        return x