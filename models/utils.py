import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def uniform_dist(a,b,size):
    std_unif = torch.rand(size)
    return std_unif*(b-a)+a

def safe_log(tens,epsilon:float=1e-5):
    return torch.log(tens+epsilon)

def generate_using_AR(model, num_images_to_generate:int):
    '''
    Generate new samples using a given autoregressive model

    Parameters
    ----------
    model: torch.nn.Module
        Trained model (MADE, PixelCNN) to be used to generate new images
    num_images_to_generate: int
        The number of images to generate
    
    Returns
    -------
    torch.FloatTensor, shape = (num_images_to_generate, self.H, self.W)
        Containing only binary values
    '''
    device = next(model.parameters()).device  # Assumes all model parameters are on the save device
    generated_imgs = torch.zeros((num_images_to_generate,model.H,model.W)).to(device)

    with torch.no_grad():
        for i in range(model.H):
            for j in range(model.W):
                pred = model.forward(generated_imgs).cpu().reshape(-1,model.H,model.W)
                generated_imgs[:,i,j] = (torch.rand(num_images_to_generate)<pred[:,i,j]).type(torch.FloatTensor).to(device)
    return generated_imgs.cpu().numpy()

def generate_using_flow(model,
                        num_samples:int=32,
                        floor:bool=True,
                        num_to_plot:int=0,
                        samples_per_row:int=5,
                        figsize=(15,8),
                        figname:str=None):
    '''
    Generate samples using a flow model

    Parameters
    ----------
    model: nn.Module, flow model

    num_samples: int
        The number of samples to generate
    floor: bool
        Whether to floor the generated samples or not
    num_to_plot: int
        Number of samples to plot using matplotlib, if 0 plotting is skipped
    samples_per_row: int
        If num_to_plot > 0, this is the number of samples to plot per row
    figsize: tuple (int, int)
        If num_to_plot > 0, this is the figsize of the matplotlib plot
    figname: str
        If not None, the matplotlib plot is saved to this location

    Returns
    -------
    np.array of floats
        shape = (num_samples, C, H, W) containing the samples generated
    '''
    x = model.sample(num_samples)

    if floor:
        x = torch.clamp(torch.floor(x),min=0,max=model.preprocess.max_val-1)
    else:
        x -= torch.min(x)
        x /= torch.max(x)

    x = x.cpu().numpy()

    if num_to_plot>0:
        num_samples = num_to_plot
        num_rows = int(np.ceil(num_samples/samples_per_row))
        fig, ax = plt.subplots(num_rows,samples_per_row,figsize=figsize)

        # Draw samples
        for i in range(num_samples):
            ax[i//samples_per_row, i%samples_per_row].imshow(x[i][0], cmap="gray")
            ax[i//samples_per_row, i%samples_per_row].set_axis_off()

        if figname is not None:
            plt.savefig(figname)
        plt.show()

    return x