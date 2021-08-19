import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .resnet import *

class Preprocessor(nn.Module):
    def __init__(self,n_bits=1,alpha=0.05):
        super().__init__()
        max_val = 2**n_bits - 1
        self.max_val = max_val + 1
        self.alpha = alpha

    def forward(self,x,invert=False):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size,...)

        invert: bool
        
        Returns
        -------
        new_x: torch.FloatTensor, shape = x.shape
            Preprocessed x
        log_det_jac: torch.FloatTensor, shape = (batch_size,)
            Log determinant of the jacobian of the preprocessing transformations done on x
        '''
        if invert:
            x = torch.sigmoid(x)  # Inverse of logit
            factor = self.max_val
            new_x = factor*(x-self.alpha)/(1-2*self.alpha)
            log_x = safe_log(x)
            log_1_x = safe_log(1-x)
            log_det_jac = log_x + log_1_x - np.log(1-2*self.alpha) + np.log(factor)

            return new_x, log_det_jac
        else:
            x = x + uniform_dist(0,1,x.shape).to(x.device)  # Dequantization
            
            # Logit Trick
            factor = (1/self.max_val)
            x = x*factor
            x = (1-2*self.alpha)*x + self.alpha
            log_x = safe_log(x)
            log_1_x = safe_log(1-x)
            new_x = log_x - log_1_x
            log_det_jac = np.log((1-2*self.alpha)*factor) - log_x - log_1_x

            return new_x, log_det_jac

class ActNorm(nn.Module):
    '''
    ActNorm normalization proposed in Glow: https://arxiv.org/abs/1807.03039

    Attributes
    ----------
    C: int
        The number of channels
    log_scale: torch.nn.Parameter, shape = (1,C,1,1)
    shift: torch.nn.Parameter, shape = (1,C,1,1)
    first_batch: bool
        True till the first pass through this layer (used for initialization on the first pass)
    '''
    def __init__(self,C:int):
        super().__init__()
        self.C = C

        self.log_scale = nn.Parameter(torch.zeros((1,self.C,1,1)))
        self.shift = nn.Parameter(torch.zeros((1,self.C,1,1)))

        self.first_batch = True

    def forward(self, x, invert:bool=False):
        if invert:
            assert not self.first_batch
            return (x - self.shift) * torch.exp(-1*self.log_scale), -1*self.log_scale*torch.ones(x.shape).to(self.shift.device)
        else:
            if self.first_batch:
                self.log_scale.data = -1*safe_log(torch.std(x,dim=(0,2,3))).reshape(1,self.C,1,1)
                self.shift.data = -1*torch.mean(x,dim=(0,2,3)).reshape(1,self.C,1,1)

                self.first_batch = False
            return x * torch.exp(self.log_scale) + self.shift, self.log_scale*torch.ones(x.shape).to(self.shift.device)

class one_one_conv(nn.Module):
    def __init__(self, num_channels, LU_decomposed=True):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, invert=False):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1]/float(c)
            if invert:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s)/float(c)

            if invert:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, invert=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, invert)
        device = next(self.parameters()).device
        ones = torch.ones(input.shape).to(device)

        if not invert:
            z = F.conv2d(input, weight)
            return z, dlogdet*ones
        else:
            z = F.conv2d(input, weight)
            return z, -1*dlogdet*ones

class one_one_conv_LU(nn.Module):
    '''
    Invertible 1x1 convolutions proposed in Glow: https://arxiv.org/abs/1807.03039
    '''
    def __init__(self, C:int):
        super().__init__()
        self.C = C
        w_init = np.random.randn(C, C)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))
        #self.weight = nn.Parameter(torch.normal(0,1,(C,C)))

    def forward(self,x,invert=False):
        B,C,H,W = x.shape
        device = next(self.parameters()).device
        ones = torch.ones(x.shape).to(device)
        log_det = torch.slogdet(self.weight)[1]/float(C)
        log_det = log_det*ones
        if invert:
            inv_weight = torch.inverse(self.weight.double()).float()
            x = F.conv2d(x, inv_weight.reshape(C,C,1,1))
            return x, -1*log_det
        else:
            x = F.conv2d(x,self.weight.reshape(C,C,1,1))
            return x, log_det

class AffineCoupling(nn.Module):
    order_type = 0
    def __init__(self,in_shape,n_filters=128,use_bn=False):
        AffineCoupling.order_type += 1
        super().__init__()

        self.mask = None
        self.use_bn = use_bn
        self.net = SimpleResnet(in_channels=in_shape[0],out_channels=2,n_filters=n_filters)
        self.scale = nn.Parameter(torch.zeros(1))
        self.scale_shift = nn.Parameter(torch.zeros(1))

    def forward(self,x,invert=False):
        assert self.mask is not None

        x_ = x*self.mask
        log_scale, t = torch.chunk(self.net(x_), 2, dim=1)
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

class AffineCouplingCheckboard(AffineCoupling):
    def __init__(self,in_shape):
        '''
        Parameters
        ----------
        in_shape: tuple[int]
            Shape of the inputs represented as (C,H,W)
        '''
        super().__init__(in_shape)
        _,H,W = in_shape

        self.mask = torch.ones(1,H,W)

        # Inefficient way to do a checkboard pattern, but it shouldn't matter as it is only in the initialization
        for h in range(H):
            counter = h%2
            for w in range(W):
                if counter%2 == 0:
                    self.mask[0][h][w] = 0
                counter += 1

        if AffineCoupling.order_type%2 == 0:
            self.mask = 1 - self.mask

        self.mask = nn.Parameter(self.mask)
        self.mask.requires_grad = False

class AffineCouplingChannel(AffineCoupling):
    def __init__(self,in_shape):
        super().__init__(in_shape)
        C, _, _ = in_shape

        self.mask = torch.ones(C,1,1)

        self.mask[:C//2] *= 0

        if AffineCoupling.order_type%2 == 0:
            self.mask = 1 - self.mask

        self.mask = nn.Parameter(self.mask.reshape(1,C,1,1))
        self.mask.requires_grad = False

class FlowSequential(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x, invert=False):
        log_det_jac = None

        layers = self.layers
        if invert:
            layers = layers[::-1]

        for layer in layers:
            x, log_det = layer.forward(x, invert=invert)
            if not type(log_det) is int and len(log_det.shape)>1:
                log_det = torch.sum(log_det,dim=(1,2,3))
            if log_det_jac is None:
                log_det_jac = log_det
            else:
                log_det_jac += log_det

        return x, log_det_jac

class Squeeze3(nn.Module):
    def __init__(self, inverted=False):
        super().__init__()
        self.inverted = inverted

    def forward(self, x, invert:bool=False):
        if self.inverted:
            invert = not invert
        if invert:
            bs,c,sl,_ = x.shape
            assert c==4

            unsqueezed = x.permute(0,2,3,1).reshape(bs,sl,sl,2,2).permute(0,1,3,2,4).reshape(bs,1,2*sl,2*sl)
            return unsqueezed, 0
        else:
            bs, c, sl, _ = x.shape
            assert c==1

            squeezed = x.reshape(bs,sl//2,2,sl//2,2).permute(0,1,3,2,4).reshape(bs,sl//2,sl//2,4).permute(0,3,1,2)
            return squeezed, 0

class Squeeze(nn.Module):
    def __init__(self, inverted=False):
        super().__init__()
        self.inverted = inverted

    def forward(self, x, invert:bool=False):
        if self.inverted:
            invert = not invert
        if invert:
            bs, c, h, w = x.shape
            unsqueezed = x.reshape(bs, c // 4, 2, 2, h, w)
            unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
            unsqueezed = unsqueezed.reshape(bs, c // 4, h * 2, w * 2)
            return unsqueezed, 0
        else:
            bs, c, h, w = x.shape

            squeezed = x.reshape(bs, c, h // 2, 2, w // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            squeezed = squeezed.reshape(bs, c * 4, h // 2, w // 2)
            return squeezed, 0
        