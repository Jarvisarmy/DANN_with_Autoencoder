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
import random
import umap
from itertools import product

from sklearn.manifold import TSNE
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def visualize_mnist(loader,size):
    temp_iter = iter(loader)
    temp_features, temp_labels = temp_iter.next()
    fig, axes = plt.subplots(size,size, figsize=(1.5*size,1.5*size))
    for i in range(size):
        for j in range(size):
            axes[i,j].imshow(temp_features[i*4+j].squeeze(),cmap='gray')
            axes[i,j].set_title(f"Label: {temp_labels[i*4+j]}")
    plt.tight_layout()
    plt.show()
    
def visualize_mnistm(loader,size):
    temp_iter = iter(loader)
    temp_features, temp_labels = temp_iter.next()
    fig, axes = plt.subplots(size,size, figsize=(1.5*size,1.5*size))
    for i in range(size):
        for j in range(size):
            axes[i,j].imshow(temp_features[i*4+j].transpose(0,2).transpose(0,1))
            axes[i,j].set_title(f"Label: {temp_labels[i*4+j]}")
    plt.tight_layout()
    plt.show()
def visualize_from_DA(autoencoder,loader,size):
    temp_iter = iter(loader)
    temp_features, temp_labels = temp_iter.next()
    temp_features = temp_features.expand(temp_features.data.shape[0],3,28,28)
    compressed_features = autoencoder(temp_features.to(device)).cpu().detach()
    #print(compressed_features.size())
    
    compressed_features = (compressed_features - compressed_features.min())*(1/(compressed_features.max()-compressed_features.min()))
    compressed_features = compressed_features.expand(compressed_features.data.shape[0],3,28,28)
    fig, axes = plt.subplots(size,size, figsize=(1.5*size,1.5*size))
    for i in range(size):
        for j in range(size):
            axes[i,j].imshow((compressed_features[i*4+j].transpose(0,2).transpose(0,1)).type(torch.float))
            axes[i,j].set_title(f"Label: {temp_labels[i*4+j]}")
    plt.tight_layout()
    plt.show()
def visualize_domain_tSNE(domain_features, domain_labels,size=None):
    if size is not None:
        perm = torch.randperm(domain_features.size(0))
        idx = perm[:size]
        domain_features = domain_features.cpu()[idx]
        domain_labels = domain_labels.cpu()[idx]
    else:
        domain_features = domain_features.cpu()
        domain_labels = domain_labels.cpu()
    tSNE = TSNE(n_components=2,
               init='random',
               perplexity=30,
               early_exaggeration=12.0,
               learning_rate=100,
               n_iter=5000,
               n_iter_without_progress=500,
               metric='euclidean',
               min_grad_norm=1e-7,
               verbose=0,
               n_jobs=-1,
               square_distances=True)
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    tSNE_embedding = tSNE.fit_transform(domain_features.cpu().detach().numpy())
    sns.scatterplot(x=tSNE_embedding[:,0],y=tSNE_embedding[:,1],hue=domain_labels.cpu().detach().numpy())
    
def visualize_domain_UMAP(domain_features, domain_labels,name="umap plot",size=None):
    sns.set_style('white')
    if size is not None:
        perm = torch.randperm(domain_features.size(0))
        idx = perm[:size]
        domain_features = domain_features.cpu()[idx]
        domain_labels = domain_labels.cpu()[idx]
    else:
        domain_features = domain_features.cpu()
        domain_labels = domain_labels.cpu()
    reducer = umap.UMAP(
                    n_components=2,
                    random_state=2022,
                    verbose=False)
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    umap_embedding = reducer.fit_transform(domain_features.cpu().detach().numpy())
    sns.scatterplot(x=umap_embedding[:,0],y=umap_embedding[:,1],hue=domain_labels.cpu().detach().numpy())
    plt.title(name)
    plt.savefig("./plots/"+name+".png")
    plt.show()