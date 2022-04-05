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
            nn.Conv2d(3,8,3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(8,16,3, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16,32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3*3*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim))

        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True),
            nn.Unflatten(dim=1,unflattened_size=(32,3,3)),
            nn.ConvTranspose2d(32,16,3,stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,3, stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,3,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x