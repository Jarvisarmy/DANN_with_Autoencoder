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

def DANNAccuracy(dann, mnist_gen, mnistm_gen):
    s_cor = 0
    t_cor = 0
    domain_cor = 0
    for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
        p = float(batch_idx)/len(mnist_gen)
        alpha = 2. / (1.+np.exp(-10*p))-1 
        source_image, source_label = source
        target_image, target_label = target
        
        source_image, source_label = source_image.to(device), source_label.to(device)
        target_image, target_label = target_image.to(device), target_label.to(device)
        
                
        source_yp_labels, source_yp_domains = dann(source_image,alpha)
        target_yp_labels, target_yp_domains = dann(target_image, alpha)
        
        source_yp_labels = source_yp_labels.data.max(1,keepdim=True)[1]
        s_cor += source_yp_labels.eq(source_label.data.view_as(source_yp_labels)).cpu().sum()
        
        target_yp_labels = target_yp_labels.data.max(1,keepdim=True)[1]
        t_cor += target_yp_labels.eq(target_label.data.view_as(target_yp_labels)).cpu().sum()
        
        source_y_domains = torch.zeros(source_label.size()[0]).type(torch.LongTensor).to(device)
        target_y_domains =  torch.ones(target_label.size()[0]).type(torch.LongTensor).to(device)
        
        source_yp_domains = source_yp_domains.data.max(1,keepdim=True)[1]
        domain_cor += source_yp_domains.eq(source_y_domains.data.view_as(source_yp_domains)).cpu().sum()
        target_yp_domains = target_yp_domains.data.max(1,keepdim=True)[1]
        domain_cor += target_yp_domains.eq(target_y_domains.data.view_as(target_yp_domains)).cpu().sum()

    domain_acc = domain_cor.item()/(len(mnist_gen.dataset)+len(mnistm_gen.dataset))
    s_acc = s_cor.item()/len(mnist_gen.dataset)
    t_acc = t_cor.item()/len(mnistm_gen.dataset)
    
    return s_acc, t_acc, domain_acc
def DANNAccuracy_source_only(dann, mnist_gen):
    s_cor = 0
    for batch_idx,source in enumerate(mnist_gen):
        p = float(batch_idx)/len(mnist_gen)
        alpha = 2. / (1.+np.exp(-10*p))-1 
        source_image, source_label = source
        
        source_image, source_label = source_image.to(device), source_label.to(device)
        
                
        source_yp_labels, source_yp_domains = dann(source_image,alpha)
        
        source_yp_labels = source_yp_labels.data.max(1,keepdim=True)[1]
        s_cor += source_yp_labels.eq(source_label.data.view_as(source_yp_labels)).cpu().sum()

    s_acc = s_cor.item()/len(mnist_gen.dataset)
    
    return s_acc

def DANNAccuracy_classify_only(dann, mnist_gen,mnistm_gen):
    s_cor = 0
    t_cor = 0
    for batch_idx,(source,target) in enumerate(zip(mnist_gen,mnistm_gen)):
        p = float(batch_idx)/len(mnist_gen)
        alpha = 2. / (1.+np.exp(-10*p))-1 
        source_image, source_label = source
        target_image, target_label = target
        source_image, source_label = source_image.to(device), source_label.to(device)
        target_image, target_label = target_image.to(device), target_label.to(device)
                
        source_yp_labels, source_yp_domains = dann(source_image,alpha)
        target_yp_labels, target_yp_domains = dann(target_image, alpha)
        source_yp_labels = source_yp_labels.data.max(1,keepdim=True)[1]
        target_yp_labels = target_yp_labels.data.max(1,keepdim=True)[1]
        s_cor += source_yp_labels.eq(source_label.data.view_as(source_yp_labels)).cpu().sum()
        t_cor += target_yp_labels.eq(target_label.data.view_as(target_yp_labels)).cpu().sum()
    s_acc = s_cor.item()/len(mnist_gen.dataset)
    t_acc = t_cor.item()/len(mnistm_gen.dataset)
    
    return s_acc, t_acc
def DANNAccuracy_with_DA(dann,autoencoder, mnist_gen, mnistm_gen):
    s_cor = 0
    t_cor = 0
    domain_cor = 0
    for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
        p = float(batch_idx)/len(mnist_gen)
        alpha = 2. / (1.+np.exp(-10*p))-1 
        source_image, source_label = source
        target_image, target_label = target
        
        source_image, source_label = source_image.to(device), source_label.to(device)
        target_image, target_label = target_image.to(device), target_label.to(device)
        
        source_image = source_image.expand(source_image.data.shape[0],3,28,28)
            
        robust_source_image = autoencoder(source_image)
        robust_target_image = autoencoder(target_image)
            
        source_image = torch.cat((source_image,robust_source_image), 1)
        target_image = torch.cat((target_image,robust_target_image),1)
        
        source_yp_labels, source_yp_domains = dann(source_image,alpha)
        target_yp_labels, target_yp_domains = dann(target_image, alpha)
      
        source_yp_labels = source_yp_labels.data.max(1,keepdim=True)[1]
        s_cor += source_yp_labels.eq(source_label.data.view_as(source_yp_labels)).cpu().sum()
        
        target_yp_labels = target_yp_labels.data.max(1,keepdim=True)[1]
        t_cor += target_yp_labels.eq(target_label.data.view_as(target_yp_labels)).cpu().sum()
        
        source_y_domains = torch.zeros(source_label.size()[0]).type(torch.LongTensor).to(device)
        target_y_domains =  torch.ones(target_label.size()[0]).type(torch.LongTensor).to(device)
        
        source_yp_domains = source_yp_domains.data.max(1,keepdim=True)[1]
        domain_cor += source_yp_domains.eq(source_y_domains.data.view_as(source_yp_domains)).cpu().sum()
        target_yp_domains = target_yp_domains.data.max(1,keepdim=True)[1]
        domain_cor += target_yp_domains.eq(target_y_domains.data.view_as(target_yp_domains)).cpu().sum()

    domain_acc = domain_cor.item()/(len(mnist_gen.dataset)+len(mnistm_gen.dataset))
    s_acc = s_cor.item()/len(mnist_gen.dataset)
    t_acc = t_cor.item()/len(mnistm_gen.dataset)
    
    return s_acc, t_acc, domain_acc
