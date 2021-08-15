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

def load_dataset(dataset_name:str,batch_size:int=16, binary:bool=False):
    '''
    Load a dataset from the following list:
    - shapes
    - mnist
    - colored_shapes
    - colored_mnist
    Downloads the dataset if needed and returns some data about the dataset (img_size, num_of_iters)
    '''
    assert dataset_name in ['mnist', 'shapes']

    existing_datasets = glob('./datasets/data/*')

    if dataset_name == 'mnist':
        dataset_path = './datasets/data/mnist.pkl'
        img_size = (1,28,28)
    elif dataset_name == 'shapes':
        dataset_path = './datasets/data/shapes.pkl'
        img_size = (1,20,20)
        
    data = np.load(dataset_path, allow_pickle=True)
    train_imgs = data['train'].transpose(0,3,1,2).astype(float)
    train_labels = data['train_labels']
    val_imgs = data['test'].transpose(0,3,1,2).astype(float)
    val_labels = data['test_labels']
    test_imgs = data['test'].transpose(0,3,1,2).astype(float)
    test_labels = data['test_labels']

    if 'colored' not in dataset_path:
        train_imgs /= 255
        val_imgs /= 255
        test_imgs /= 255

    if binary:
        train_imgs = np.round(train_imgs).astype(float)
        val_imgs = np.round(val_imgs).astype(float)
        test_imgs = np.round(test_imgs).astype(float)

    train_dataset = ImgDataset(train_imgs, train_labels)
    val_dataset = ImgDataset(val_imgs, val_labels)
    test_dataset = ImgDataset(test_imgs, test_labels)

    train_loader = iter_wrapper(train_dataset, batch_size)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    return img_size, train_loader, val_loader, test_loader