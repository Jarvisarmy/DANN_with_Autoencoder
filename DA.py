import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class DenoisingAutoencoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3,kernel_size=3,padding=1),
            # 50 x 8 x 8
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Sigmoid()
            #nn.Flatten()
        )

        self.decoder = nn.Sequential(
            #nn.Linear(encoded_space_dim, 128),
            #nn.ReLU(True),
            #nn.Linear(128, 3*3*32),
            #nn.ReLU(True),
            #nn.Unflatten(dim=1,unflattened_size=(16,4,4)),
            #nn.ConvTranspose2d(32,16,3,stride=2, output_padding=0),
            #nn.BatchNorm2d(16),
            #nn.ReLU(True),
            nn.ConvTranspose2d(3,3,3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3,3,3,padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.expand(x.data.shape[0],3,28,28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x