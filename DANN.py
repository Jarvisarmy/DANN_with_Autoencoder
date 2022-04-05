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
    
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,padding=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        #x = self.bn1(x)
        x = self.max1(x)
        
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu2(x)
        x = self.max2(x)
        
        x = x.view(-1,7*7*48)
        return x
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(in_features=7*7*48, out_features=100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1= nn.ReLU()
        self.linear2 = nn.Linear(in_features=100, out_features = 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2= nn.ReLU()
        self.linear3 = nn.Linear(in_features=100, out_features = 10)
        #self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        #x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = nn.functional.softmax(x,dim=1)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(in_features=7*7*48, out_features=100)
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
        #x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        #x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        #return torch.nn.LogSoftmax(x,dim=1)
        return x