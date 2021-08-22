import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *

class VAE(nn.Module):
    def __init__(self, in_shape, n_bits=1):
        super().__init__()
        self.in_shape = in_shape
        self.n_bits = n_bits
        self.n_vals = 2**n_bits

        self.encoder = 

        self.decoder = 
        
    
    def forward(self, x):
        pass

    def loss(self, x):
        pass

    def sample(self, num_samples:int, temp:float=0.7):
        pass

class VQ_VAE(nn.Module):
    def __init__(self, in_shape, n_bits=1):
        super().__init__()
        self.in_shape = in_shape
        self.n_bits = n_bits
        self.n_vals = 2**n_bits
    def forward(self, x):
        pass

    def loss(self, x):
        pass

    def sample(self, num_samples:int, temp:float=0.7):
        pass