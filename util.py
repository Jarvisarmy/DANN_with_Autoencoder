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
from DA import DenoisingAutoencoder
from tsne_torch import TorchTSNE as TSNE
from DANN import DANN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_domain_datas(mnist_gen,mnistm_gen):
    total_x = torch.cat([torch.stack([torch.cat((x[0],x[0],x[0]),0) for x in mnist_gen.dataset],axis=0).view(60000,-1),
                  torch.stack([x[0] for x in mnistm_gen.dataset],axis=0).view(60000,-1)],axis=0).to(device)
    total_y = torch.cat([torch.tensor([1 for x in mnist_gen.dataset]),
                  torch.tensor([0 for x in mnistm_gen.dataset])],axis=0).to(device)
    return total_x, total_y

def generate_domain_datas_from_dann(dann, mnist_gen,mnistm_gen):
    with torch.no_grad():
        total_x = torch.tensor([])
        total_y = torch.tensor([])
        for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
            source_image, source_label = source
            target_image, target_label = target

            # the source is 1 * 28 * 28, we have to preprocess it
            #source_image = torch.cat((source_image, source_image, source_image),1)

            domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                        torch.ones(target_label.size()[0]).type(torch.LongTensor)),0)
            source = source_image.expand(source_image.data.shape[0],3,28,28)
            source = dann.extractor(source.to(device))
            source = source.view(-1,50*4*4).cpu()
            target = target_image.expand(target_image.data.shape[0],3,28,28)
            target = dann.extractor(target.to(device))
            target = target.view(-1,50*4*4).cpu()
            temp = torch.cat((source,target),0)
            total_x = torch.cat((total_x,temp),0)
            total_y = torch.cat((total_y,domain_label),0)
    return total_x, total_y
def generate_domain_datas_from_extractor_with_DA(dann,autoencoder, mnist_gen,mnistm_gen):
    with torch.no_grad():
        total_x = torch.tensor([])
        total_y = torch.tensor([])
        for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
            source_image, source_label = source
            target_image, target_label = target
            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image, target_label = target_image.to(device), target_label.to(device)

            # the source is 1 * 28 * 28, we have to preprocess it
            #source_image = torch.cat((source_image, source_image, source_image),1)

            domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                        torch.ones(target_label.size()[0]).type(torch.LongTensor)),0)
            source = source_image.expand(source_image.data.shape[0],3,28,28)
            robust_source = autoencoder(source.to(device))
            source = torch.cat((source,robust_source), 1)
            source = dann.extractor(source).cpu()
            source = source.view(-1,50*4*4)
            target = target_image.expand(target_image.data.shape[0],3,28,28)
            robust_target = autoencoder(target.to(device))
            target = torch.cat((target,robust_target), 1)
            target = dann.extractor(target).cpu()
            target = target.view(-1,50*4*4)
            temp = torch.cat((source,target),0)
            total_x = torch.cat((total_x,temp),0)
            total_y = torch.cat((total_y,domain_label),0)
    return total_x, total_y
def generate_domain_datas_DA(autoencoder, mnist_gen,mnistm_gen):
    with torch.no_grad():
        total_x = torch.tensor([])
        total_y = torch.tensor([])
        for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
            source_image, source_label = source
            target_image, target_label = target

            # the source is 1 * 28 * 28, we have to preprocess it
            #source_image = torch.cat((source_image, source_image, source_image),1)

            domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                        torch.ones(target_label.size()[0]).type(torch.LongTensor)),0)
            source = source_image.expand(source_image.data.shape[0],3,28,28)
            
            source = autoencoder(source.to(device))
            #source = source.expand(source.data.shape[0],3,28,28).cpu()
            #source = (source - source.min())*(1/(source.max()-source.min()))
            #source = dann.extractor(source.to(device))
            source = source.view(-1,3*28*28).cpu()
            target = target_image.expand(target_image.data.shape[0],3,28,28)
            target = autoencoder(target.to(device))
            #target = target.expand(target.data.shape[0],3,28,28).cpu()
            #target = (target - target.min())*(1/(target.max()-target.min()))
            #target = dann.extractor(target.to(device))
            target = target.view(-1,3*28*28).cpu()
            temp = torch.cat((source,target),0)
            total_x = torch.cat((total_x,temp),0)
            total_y = torch.cat((total_y,domain_label),0)
    return total_x, total_y
def save_model(model, path):
    torch.save(model.state_dict(),path)

def load_DANN(name,isGPU=True):
    dann = DANN()
    if (isGPU):
        dann = dann.to(device)
    dann.load_state_dict(torch.load(name))
    return dann
def load_DA(name,isGPU=True):
    autoencoder = DenoisingAutoencoder(100)
    if (isGPU):
        autoencoder = autoencoder.to(device)
    autoencoder.load_state_dict(torch.load(name))
    return autoencoder
def add_noise(inputs, corruption_prob):
    noisy_input = inputs + torch.randn_like(inputs)*corruption_prob
    noisy_input = torch.clip(noisy_input,0.,1.)
    return noisy_input

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum=momentum
        self.reset()
        
    def reset(self):
        self.val=None
        self.avg = 0
    def update(self, val):
        if self.val is None:
            self.avg=val
        else:
            self.avg = self.avg*self.momentum + val*(1-self.momentum)
        self.val = val
        
        