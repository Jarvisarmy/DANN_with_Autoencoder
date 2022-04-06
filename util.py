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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_domain_datas(mnist_gen,mnistm_gen):
    total_x = torch.cat([torch.stack([torch.cat((x[0],x[0],x[0]),0) for x in mnist_gen.dataset],axis=0).view(60000,-1),
                  torch.stack([x[0] for x in mnistm_gen.dataset],axis=0).view(60000,-1)],axis=0).to(device)
    total_y = torch.cat([torch.tensor([1 for x in mnist_gen.dataset]),
                  torch.tensor([0 for x in mnistm_gen.dataset])],axis=0).to(device)
    return total_x, total_y

def generate_domain_datas_from_extractor(extractor, mnist_gen,mnistm_gen):
    with torch.no_grad():
        total_x = torch.tensor([])
        total_y = torch.tensor([])
        for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
            source_image, source_label = source
            target_image, target_label = target

            # the source is 1 * 28 * 28, we have to preprocess it
            source_image = torch.cat((source_image, source_image, source_image),1)
            total_image = torch.cat((source_image, target_image), 0)
            domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                        torch.ones(target_label.size()[0]).type(torch.LongTensor)),0)
            
            temp = extractor(total_image.to(device)).cpu()
            total_x = torch.cat((total_x,temp),0)
            total_y = torch.cat((total_y,domain_label),0)
    return total_x, total_y

def save_DANN(extractor, classifier, discriminator):
    torch.save(extractor.state_dict(),"./models/extractor.pt")
    torch.save(classifier.state_dict(),"./models/classifier.pt")
    torch.save(discriminator.state_dict(),"./models/discriminator.pt")
def save_DA(autoencoder,name):
    torch.save(autoencoder.state_dict(),name)

def load_DANN(isGPU=True):
    extractor = FeatureExtractor()
    classifier = Classifier()
    discriminator = Discriminator()
    if (isGPU):
        extractor = extractor.to(device)
        classifier = classifier.to(device)
        discriminator = discriminator.to(device)
    extractor.load_state_dict(torch.load("./models/extractor.pt"))
    classifier.load_state_dict(torch.load("./models/classifier.pt"))
    discriminator.load_state_dict(torch.load("./models/discriminator.pt"))
    return extractor, classifier, discriminator
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
        
        