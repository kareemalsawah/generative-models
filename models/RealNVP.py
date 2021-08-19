import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *
from .Flows import *

class RealNVP(nn.Module):
    '''
    Implemented of RealNVP proposed in https://arxiv.org/abs/1605.08803

    Attributes
    ----------
    in_shape: tuple (int, int, int)
        The shape of images as inputs (C, H, W)
    z_dist: nn.Module (another tractable generative model) or torch.distribution.Distribution
        The base tractable distribution for the latent variable Z
    max_val: int
    large_model: bool
        If True, a larger model is used
    '''
    def __init__(self,in_shape,z_dist,n_bits=1,large_model=False):
        super().__init__()
        self.in_shape = in_shape
        self.z_dist = z_dist
        self.n_bits = n_bits
        self.is_z_simple = isinstance(self.z_dist, torch.distributions.Distribution)
        self.large_model = large_model

        preprocess = Preprocessor(n_bits)

        layers_1 = self.create_checkboard(in_shape, 3)

        scale_2_shape = (in_shape[0]*4,in_shape[1]//2,in_shape[2]//2)
        layers_2 = self.create_channel(scale_2_shape, 3)

        if self.large_model:
            layers_3 = self.create_checkboard(in_shape, 3)
            layers_4 = self.create_channel(scale_2_shape, 3)
            layers_5 = self.create_checkboard(scale_2_shape, 4)
            self.layers = FlowSequential([preprocess,
                                            layers_1,
                                            Squeeze(False),
                                            layers_2,
                                            Squeeze(True),
                                            layers_3,
                                            Squeeze(False),
                                            layers_4,
                                            Squeeze(True),
                                            layers_5])
        else:
            layers_3 = self.create_checkboard(scale_2_shape, 4)
            self.layers = FlowSequential([preprocess,
                                            layers_1,
                                            Squeeze(False),
                                            layers_2,
                                            Squeeze(True),
                                            layers_3])
    
    def create_checkboard(self, in_shape, num_blocks:int, add_act_norm=True):
        blocks = []
        for i in range(num_blocks):
            blocks.append(AffineCouplingCheckboard(in_shape))
            if i > num_blocks-1 and add_act_norm:
                blocks.append(ActNorm(in_shape[0]))
        
        return FlowSequential(blocks)
    
    def create_channel(self, in_shape, num_blocks:int, add_act_norm=True):
        blocks = []
        for i in range(num_blocks):
            blocks.append(AffineCouplingChannel(in_shape))
            if i > num_blocks-1 and add_act_norm:
                blocks.append(ActNorm(in_shape[0]))
        
        return FlowSequential(blocks)

    def forward(self,x,invert=False):
        x, log_det_jac = self.layers.forward(x, invert=invert)
        return x, log_det_jac

    def log_prob(self, x, invert=False):
        num_dims = self.in_shape[0]*self.in_shape[1]*self.in_shape[2]

        if self.is_z_simple:
            assert invert == False

        x, log_det_jac = self.forward(x, invert=invert)
        if self.is_z_simple:
            log_prob_z = self.z_dist.log_prob(x)
            log_prob_z = torch.sum(log_prob_z,dim=(1,2,3))
        else:
            log_prob_z = self.z_dist.log_prob(x, invert=invert)

        log_prob_x = log_prob_z + log_det_jac
        return log_prob_x/num_dims

    def loss(self, x):
        nll = -1*self.log_prob(x).mean()
        return nll
    
    def sample(self, num_samples:int):
        device = next(self.parameters()).device

        z = self.z_dist.sample((num_samples,*self.in_shape)).to(device)

        self.eval()
        with torch.no_grad():
            x, _ = self.forward(z, invert=True)
        
        return x