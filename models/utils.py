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

def generate_using_AR(model, num_images_to_gen:int):
    '''
    '''
    def sample_dist(probs):
        '''
        Parameters
        ----------
        probs: numpy.float array, shape = (-1, n)
        '''
        n = probs.shape[1]
        generated = []
        for prob in probs:
            generated.append(np.random.choice(np.arange(n), p=prob))
        return np.array(generated)

    images = torch.zeros((num_images_to_gen, model.C, model.H, model.W))
    device = next(model.parameters()).device
    images = images.to(device)
    max_val = int(2**model.n_bits)

    H, W, C = model.H, model.W, model.C

    with torch.no_grad():
        model.eval()
        for i in range(H):
            for j in range(W):
                logits = model.forward(images)
                probs = torch.exp(logits).reshape(num_images_to_gen, C, H, W, max_val)[:, :, i, j].reshape(-1, max_val)
                probs = probs.cpu().numpy()

                sampled = sample_dist(probs)

                sampled = sampled.reshape(num_images_to_gen, C)
                images[:, :, i, j] = torch.from_numpy(sampled).to(device)

    model.train()
    return images.cpu().numpy()/float(max_val-1)

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