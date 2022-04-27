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
    def __init__(self,out_dim=50*4*4):
        super(DANN,self).__init__()
        self.out_dim = out_dim
        self.extractor = nn.Sequential(
            # 3 x 28 x 28
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 24 x 24
            nn.MaxPool2d(kernel_size=2),
            # 64 x 12 x 12
            nn.Conv2d(in_channels=64, out_channels=50,kernel_size=5),
            # 50 x 8 x 8
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2))
            # 50 x 4 x 4
        self.classifier = nn.Sequential(
            nn.Linear(in_features = out_dim, out_features = 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=100, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features=10),
            nn.LogSoftmax(dim=1))
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
            nn.LogSoftmax(dim=1))
        
        
    def forward(self,x,alpha):
        x = x.expand(x.data.shape[0],3,28,28)
        x = self.extractor(x)
        x = x.view(-1,self.out_dim)
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

class DANN_v2(nn.Module):
    def __init__(self,out_dim=50*4*4):
        super(DANN_v2,self).__init__()
        self.out_dim = out_dim
        self.extractor = nn.Sequential(
            # 3 x 28 x 28
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 24 x 24
            nn.MaxPool2d(kernel_size=2),
            # 64 x 12 x 12
            nn.Conv2d(in_channels=64, out_channels=50,kernel_size=5),
            # 50 x 8 x 8
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2))
            # 50 x 4 x 4
            #nn.Linear(in_features = 6*28*28, out_features = 1024),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(in_features = 1024, out_features = 648))
        self.classifier = nn.Sequential(
            nn.Linear(in_features = out_dim, out_features = 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=100, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features=10),
            nn.LogSoftmax(dim=1))
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
            nn.LogSoftmax(dim=1))
        
        
    def forward(self,x,alpha):
        #x = x.expand(x.data.shape[0],3,28,28)
        #x = x.view(-1,6*28*28)
        x = self.extractor(x)
        x = x.view(-1,self.out_dim)
        rev_x = ReverseLayerF.apply(x,alpha)
        labels = self.classifier(x)
        domains = self.discriminator(rev_x)
        return labels, domains
