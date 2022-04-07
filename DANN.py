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
    
class ConvolutionalExtractor(nn.Module):
    def __init__(self):
        super(ConvolutionalExtractor,self).__init__()
        # 3 x28 x 28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1)
        
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        # 32 x 28 x 28
        self.max1 = nn.MaxPool2d(kernel_size=2)
        # 32 x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3,padding=1)
        # 48 x 14 x 14
        
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(48)
        # 48 x 7 x 7
        self.max2 = nn.MaxPool2d(kernel_size=2)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.max1(x)
        
        x = self.conv2(x)
        
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.max2(x)
        
        x = x.view(-1,3*28*28)
        return x

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

class Classifier(nn.Module):
    def __init__(self,in_features):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1= nn.ReLU()
        self.linear2 = nn.Linear(in_features=100, out_features = 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2= nn.ReLU()
        self.linear3 = nn.Linear(in_features=100, out_features = 10)
        #self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.linear1(x)
        
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.linear2(x)
        
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.linear3(x)
        x = nn.functional.softmax(x,dim=1)
        return x
    
class Discriminator(nn.Module):
    def __init__(self,in_features):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=100, out_features=100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=100, out_features=2)
        #self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x,alpha):
        x = ReverseLayerF.apply(x,alpha)
        x = self.linear1(x)
        
        x = self.relu1(x)
        x = self.bn1(x)
        #x = self.linear2(x)
        #x = self.bn2(x)
        #x = self.relu2(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        #return torch.nn.LogSoftmax(x,dim=1)
        return x