import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from zipfile import ZipFile
import skimage.io
from PIL import Image
import seaborn as sns

from tsne_torch import TorchTSNE as TSNE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_mnist_loaders(data_aug = False, batch_size=128,test_batch_size=1000):
    if data_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28,padding=4),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=True, download=True,transform=train_transform),batch_size=batch_size, shuffle=True,drop_last=False)
    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=True, download=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=False, download=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=False)
    return train_loader, train_eval_loader, test_loader