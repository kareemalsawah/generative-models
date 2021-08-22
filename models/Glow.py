import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .utils import *
from .Flows import *

class GlowAffineCoupling(nn.Module):
    def __init__(self, in_channels, filter_size=512):
        super().__init__()

        self.flow_net = nn.Sequential(nn.Conv2d(in_channels, filter_size, kernel_size=(3,3), padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(filter_size, filter_size, kernel_size=(1,1), padding=0),
                                        nn.ReLU(),
                                        nn.Conv2d(filter_size, in_channels*2, kernel_size=(3,3), padding=1))

        self.flow_net[-1].weight.data *= 0
        self.flow_net[-1].bias.data *= 0

        self.mask = torch.ones(in_channels,1,1)

        self.mask[:in_channels//2] *= 0

        self.mask = nn.Parameter(self.mask.reshape(1,in_channels,1,1))
        self.mask.requires_grad = False

        self.scale = nn.Parameter(torch.zeros(1))
        self.scale_shift = nn.Parameter(torch.zeros(1))


    def forward(self, x, invert=False):
        x_ = x*self.mask
        log_scale, t = torch.chunk(self.flow_net(x_), 2, dim=1)
        log_scale = self.scale*torch.tanh(log_scale) + self.scale_shift

        t = t * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        if invert:
            z = (x - t) / torch.exp(log_scale)
            log_det_jacobian = -1*log_scale
        else:
            z = x * torch.exp(log_scale) + t
            log_det_jacobian = log_scale

        return z, log_det_jacobian

class GlowStep(nn.Module):
    def __init__(self,in_shape,filter_size=512):
        super().__init__()
        in_channels = in_shape[0]
        actnorm = ActNorm(in_channels)
        conv_perm = one_one_conv(in_channels)
        coupling = GlowAffineCoupling(in_channels, filter_size)
        self.layers = FlowSequential([actnorm, conv_perm, coupling])
    
    def forward(self, x, invert=False):
        return self.layers.forward(x, invert=invert)

class GlowBlockSqueeze(nn.Module):
    def __init__(self, n_blocks, in_shape, filter_size=512):
        super().__init__()
        blocks = []
        scale_2_shape = (in_shape[0]*4,in_shape[1]//2,in_shape[2]//2)
        blocks.append(Squeeze(False))
        for _ in range(n_blocks):
            blocks.append(GlowStep(scale_2_shape,filter_size))
        self.layers = FlowSequential(blocks)
    
    def forward(self, x, invert=False):
        return self.layers.forward(x, invert=invert)
    

class Glow(nn.Module):
    def __init__(self,in_shape,z_dist,n_blocks,flows_per_block,n_bits=1):
        super().__init__()
        self.in_shape = in_shape
        self.z_dist = z_dist
        self.is_z_simple = isinstance(self.z_dist, torch.distributions.Distribution)
        preprocess = Preprocessor(n_bits)
        self.n_bits = n_bits
        
        layers = [preprocess]
        for _ in range(n_blocks):
            layers.append(GlowBlockSqueeze(flows_per_block, in_shape))
            in_shape = in_shape[0]*4, in_shape[1]//2, in_shape[2]//2
        for _ in range(n_blocks):
            layers.append(Squeeze(True))
        self.layers = FlowSequential(layers)

    def forward(self,x,invert=False):
        return self.layers.forward(x, invert=invert)


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
    
    def sample(self, num_samples:int, temp:float=0.7):
        device = next(self.parameters()).device

        z = self.z_dist.sample((num_samples,*self.in_shape)).to(device)*temp

        self.eval()
        with torch.no_grad():
            x, _ = self.forward(z, invert=True)
        
        return x


class GlowMultiScale(nn.Module):
    def __init__(self,in_shape,z_dist,n_blocks,flows_per_block,n_bits=1):
        super().__init__()
        self.in_shape = in_shape
        self.z_dist = z_dist
        self.is_z_simple = isinstance(self.z_dist, torch.distributions.Distribution)
        preprocess = Preprocessor(n_bits)
        self.n_bits = n_bits
        
        layers = [preprocess]
        self.z_shapes = []
        for _ in range(n_blocks):
            layers.append(GlowBlockSqueeze(flows_per_block, in_shape))
            in_shape = in_shape[0]*2, in_shape[1]//2, in_shape[2]//2
            self.z_shapes.append(in_shape)
        
        self.z_shapes.append(in_shape)

        self.layers = FlowSequential(layers)


    def forward(self,x,invert=False):
        if invert:
            z_list = x[::-1]
            log_det_jac = 0
            z = z_list[0]
            curr_idx = 1
            for layer in self.layers.layers[::-1]:
                if type(layer) is GlowBlockSqueeze:
                    z = torch.cat([z,z_list[curr_idx]],dim=1)
                    curr_idx += 1

                z, log_det = layer.forward(z, invert=True)
                if len(log_det.shape)>1:
                    log_det = torch.sum(log_det, dim=(1,2,3))
                log_det_jac += log_det
            return z, log_det_jac
        else:
            z = x
            z_combined = []
            log_det_jac = 0
            for layer in self.layers.layers:
                z, log_det = layer.forward(z)
                log_det_jac += log_det
                if type(layer) is GlowBlockSqueeze:
                    z, z_1 = torch.chunk(z, 2, dim=1)
                    z_combined.append(z_1)
            z_combined.append(z)

        return z_combined, log_det_jac


    def log_prob(self, x, invert=False):
        num_dims = self.in_shape[0]*self.in_shape[1]*self.in_shape[2]
        if self.is_z_simple:
            assert invert == False

        x_combined, log_det_jac = self.forward(x, invert=invert)
        if self.is_z_simple:
            log_prob_z = 0
            for z in x_combined:
                log_prob = self.z_dist.log_prob(z)
                log_prob_z += torch.sum(log_prob,dim=(1,2,3))
            log_prob_x = log_prob_z + log_det_jac
            return log_prob_x/num_dims

        if invert:
            log_prob_x, log_det = self.z_dist.log_prob(x, invert=True)
            log_prob_z = log_prob_x + log_det + log_det_jac
            return log_prob_z/num_dims
        else:
            log_prob_z, log_det = self.z_dist.log_prob(x, invert=False)
            log_prob_x = log_prob_z + log_det + log_det_jac
            return log_prob_x/num_dims
    
    def loss(self, x):
        nll = -1*self.log_prob(x).mean()
        return nll
    
    def sample(self, num_samples:int, temp:float=0.7):
        device = next(self.parameters()).device

        z_list = []
        for z_shape in self.z_shapes:
            z = self.z_dist.sample((num_samples,*z_shape)).to(device)*temp
            z_list.append(z)

        self.eval()
        with torch.no_grad():
            x, _ = self.forward(z_list, invert=True)
        
        return x