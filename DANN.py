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
from sklearn.manifold import TSNE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.alpha * grad_output.neg()
        return output, None
    
class DANN(nn.Module):
    def __init__(self):
        super(ConvolutionalExtractor,self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=50,kernel_size=5),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 50*4*4, out_features = 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
            nn.Linear(in_features=100, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features=10),
            nn.LogSoftmax(dim=1))
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=50*4*4, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
            nn.LogSoftmax(dim=1))
        
        
    def forward(self,x,alpha):
        x = x.expand(x.data.shape[0],3,28,28)
        x = self.extractor(x)
        x = x.view(-1,50*4*4)
        rev_x = ReverseLayerF.apply(x,alpha)
        labels = self.classifier(x)
        domains = self.discriminator(rev_x)
        return labels, domains


class LinearExtractor(nn.Module):
    def __init__(self,in_features):
        super(LinearExtractor,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=258),
            nn.ReLU(),
            nn.BatchNorm1d(258),
            nn.Linear(in_features=258,out_features=258)
        )
    def forward(self,x):
        x = self.linear(x)
        return x