import torch
import torch.nn as nn
import numpy as np


def eval_model(model, test_loader):
    '''
    Evaluate a given model on a test or validation set

    Parameters
    ----------
    model: nn.Module with a function loss that takes imgs as input
    test_loader: torch DataLoader

    Returns
    -------
    float: average test loss over all test examples
    '''
    device = next(model.parameters()).device
    test_loss = 0
    num_test = 0

    model.eval()
    with torch.no_grad():
        for imgs,lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            loss = model.loss(imgs)
            test_loss += loss.item()*imgs.shape[0]
            num_test += imgs.shape[0]
    
    model.train()
    return test_loss/num_test
