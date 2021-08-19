import torch
import torch.nn as nn
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader


class ImgDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(np.array(images,dtype=np.float)).type(torch.FloatTensor)
        self.labels = torch.from_numpy(np.array(labels,dtype=int)).type(torch.LongTensor)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def iter_wrapper(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loader = iter(data_loader)

    while True:
        try:
            yield next(data_loader)

        except StopIteration:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_loader = iter(data_loader)
            yield next(data_loader)

def load_dataset(dataset_name:str,batch_size:int=16, n_bits:int=1):
    '''
    Load a dataset from the following list:
    - shapes
    - mnist
    - colored_shapes
    - colored_mnist
    Downloads the dataset if needed and returns some data about the dataset (img_size, num_of_iters)
    '''
    assert dataset_name in ['mnist', 'shapes','colored_mnist','colored_shapes','celeba_32']

    if dataset_name  == 'mnist':
        img_size = (1,28,28)
    elif dataset_name == 'shapes':
        img_size = (1,20,20)
    elif dataset_name == 'colored_mnist':
        img_size = (3,28,28)
    elif dataset_name == 'colored_shapes':
        img_size = (3,20,20)
    elif dataset_name == 'celeba_32':
        img_size = (3,32,32)
    
    dataset_path = './datasets/data/{}.pkl'.format(dataset_name)
    data = np.load(dataset_path, allow_pickle=True)
    train_imgs = data['train']
    train_labels = data['train_labels']
    val_imgs = data['test']
    val_labels = data['test_labels']
    test_imgs = data['test']
    test_labels = data['test_labels']

    # Set Number of bits
    max_val = 2**n_bits - 1
    train_imgs = np.round(train_imgs*max_val).astype(np.float)
    val_imgs = np.round(val_imgs*max_val).astype(np.float)
    test_imgs = np.round(test_imgs*max_val).astype(np.float)

    train_dataset = ImgDataset(train_imgs, train_labels)
    val_dataset = ImgDataset(val_imgs, val_labels)
    test_dataset = ImgDataset(test_imgs, test_labels)

    train_loader = iter_wrapper(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    return img_size, train_loader, val_loader, test_loader