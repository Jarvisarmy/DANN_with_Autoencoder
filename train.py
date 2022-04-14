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
from test import *
from DANN import *
import torch.optim as optim
from DA import DenoisingAutoencoder
from util import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def DANNTrain(mnist_train, mnistm_train, mnist_eval, mnistm_eval, epochs):
    dann = DANN().to(device)

    criterion= nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        list(dann.parameters()),lr=0.01)
    domain_accs =[]
    source_accs = []
    target_accs = []
    total_steps = epochs* len(mnistm_train)
    for epoch in range(epochs):
        start_steps = epoch * len(mnist_train)
        
        for batch_idx, (source, target) in enumerate(zip(mnist_train, mnistm_train)):
            source_image, source_label = source
            target_image, target_label = target

            p = float(batch_idx + start_steps)/ total_steps
            alpha = 2. / (1.+np.exp(-10*p))-1 

            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image, target_label = target_image.to(device), target_label.to(device)

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01/(1.+10*p)**0.75

            # clear the grad
            optimizer.zero_grad()
            
            source_yp_labels, source_yp_domains = dann(source_image,alpha)
            target_yp_labels, target_yp_domains = dann(target_image,alpha)

            
            source_labels_loss = criterion(source_yp_labels, source_label)
            source_domain_loss = criterion(source_yp_domains, torch.zeros(source_label.size()[0]).type(torch.LongTensor).to(device))
            target_domain_loss = criterion(target_yp_domains, torch.ones(target_label.size()[0]).type(torch.LongTensor).to(device))
            
            total_loss = source_labels_loss + source_domain_loss + target_domain_loss
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            source_acc, target_acc, domain_acc = DANNAccuracy(dann, mnist_eval, mnistm_eval)
            domain_accs.append(domain_acc)
            source_accs.append(source_acc)
            target_accs.append(target_acc)
            print(f'{epoch+1}/{epochs}: source_acc: {source_acc},target_acc: {target_acc}, domain_acc: {domain_acc}')
    return source_accs, target_accs, domain_accs, dann
            
def DANNTrain_source_only(mnist_train,mnist_eval,epochs):
    dann = DANN().to(device)

    criterion= nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        list(dann.parameters()),lr=0.01)
    source_accs = []
    total_steps = epochs* len(mnist_train)
    for epoch in range(epochs):
        start_steps = epoch * len(mnist_train)
        
        for batch_idx, source in enumerate(mnist_train):
            source_image, source_label = source

            p = float(batch_idx + start_steps)/ total_steps
            alpha = 2. / (1.+np.exp(-10*p))-1 

            source_image, source_label = source_image.to(device), source_label.to(device)

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01/(1.+10*p)**0.75

            # clear the grad
            optimizer.zero_grad()
            
            source_yp_labels, source_yp_domains = dann(source_image,alpha)

            
            source_labels_loss = criterion(source_yp_labels, source_label)
           
            total_loss = source_labels_loss
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            source_acc  = DANNAccuracy_source_only(dann, mnist_eval)
            source_accs.append(source_acc)
            print(f'{epoch+1}/{epochs}: source_acc: {source_acc}')
    return source_accs, dann
    
def DATrain(mnist_train, mnistm_train, mnist_eval, mnistm_eval, encoded_space_dim, epochs):
    autoencoder = DenoisingAutoencoder(encoded_space_dim).to(device)
    lossf = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.01)
    loss_arr = []
    for epoch in range(epochs):
        for batch_idx, (source, target) in enumerate(zip(mnist_train, mnistm_train)):
            source_image, source_label = source
            target_image, target_label = target

                # the source is 1 * 28 * 28, we have to preprocess it
            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image, target_label = target_image.to(device), target_label.to(device)

            # clear the grad
            optimizer.zero_grad()
            source_image = add_noise(source_image,0.3)
            target_image = add_noise(target_image,0.3)
            reconstructed_source = autoencoder(source_image)
            reconstructed_target = autoencoder(target_image)

            loss_source = lossf(reconstructed_source, source_image.expand(source_image.shape[0],3,28,28))
            loss_target = lossf(reconstructed_target, target_image)
            loss = loss_source+loss_target
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (source, target) in enumerate(zip(mnist_eval, mnistm_eval)):
                source_image, source_label = source
                target_image, target_label = target


                source_image, source_label = source_image.to(device), source_label.to(device)
                target_image, target_label = target_image.to(device), target_label.to(device)

                reconstructed_source = autoencoder(source_image)
                reconstructed_target = autoencoder(target_image)

                loss_source = lossf(reconstructed_source, source_image.expand(source_image.shape[0],3,28,28))
                loss_target = lossf(reconstructed_target, target_image)
                loss = loss_source+loss_target
                total_loss += loss
            loss_arr.append(total_loss.cpu()/(2*len(mnist_eval)))
            print(f'{epoch+1}/{epochs}: avg_loss:{total_loss/(2*len(mnist_eval))}')
    return loss_arr, autoencoder


    