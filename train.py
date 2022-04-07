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
    extractor = ConvolutionalExtractor().to(device)
    classifier = Classifier(3*28*28).to(device)
    discriminator = Discriminator(3*28*28).to(device)
    for p in extractor.parameters():
        p.requires_grad = True
    for p in classifier.parameters():
        p.requires_grad = True
    for p in discriminator.parameters():
        p.requires_grad = True

    class_lossf= nn.NLLLoss().to(device)
    domain_lossf= nn.NLLLoss().to(device)
    optimizer = optim.SGD(
        list(extractor.parameters())+
        list(classifier.parameters())+
        list(discriminator.parameters()),lr=0.01,momentum=0.8)
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

            # the source is 1 * 28 * 28, we have to preprocess it
            source_image = torch.cat((source_image, source_image, source_image),1)
            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image, target_label = target_image.to(device), target_label.to(device)
            total_image = torch.cat((source_image, target_image), 0)
            domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                    torch.ones(target_label.size()[0]).type(torch.LongTensor)),0).to(device)

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01/(1.+10*p)**0.75

            # clear the grad
            optimizer.zero_grad()

            total_feature = extractor(total_image)
            source_feature = extractor(source_image)

            # classification loss
            yp = classifier(source_feature)
            class_loss = class_lossf(yp,source_label)

            # domain discriminate loss
            domain_yp = discriminator(total_feature,1)
            domain_loss = domain_lossf(domain_yp,domain_label)

            total_loss = class_loss+domain_loss
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            source_acc, target_acc, domain_acc = DANNAccuracy(extractor, classifier, discriminator, mnist_eval, mnistm_eval)
            domain_accs.append(domain_acc)
            source_accs.append(source_acc)
            target_accs.append(target_acc)
            print(f'{epoch+1}/{epochs}: source_acc: {source_acc},target_acc: {target_acc}, domain_acc: {domain_acc}')
    return source_accs, target_accs, domain_accs, extractor, classifier, discriminator
            
def DANNTrain_source_only(mnist_train,mnist_eval,epochs):
    extractor = ConvolutionalExtractor().to(device)
    classifier = Classifier(3*28*28).to(device)
    class_lossf= nn.NLLLoss().to(device)
   
    optimizer = optim.SGD(
        list(extractor.parameters())+
        list(classifier.parameters()),lr=0.01,momentum=0.8)

    source_accs = []

    total_steps = epochs* len(mnist_train)
    for epoch in range(epochs):
        start_steps = epoch * len(mnist_train)
        
        for batch_idx, source in enumerate(mnist_train):
            source_image, source_label = source

            p = float(batch_idx + start_steps)/ total_steps
            alpha = 2. / (1.+np.exp(-10*p))-1 

            # the source is 1 * 28 * 28, we have to preprocess it
            source_image = torch.cat((source_image, source_image, source_image),1)
            source_image, source_label = source_image.to(device), source_label.to(device)

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01/(1.+10*p)**0.75

            # clear the grad
            optimizer.zero_grad()

            source_feature = extractor(source_image)

            # classification loss
            yp = classifier(source_feature)
            class_loss = class_lossf(yp,source_label)

            total_loss = class_loss
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            source_acc = DANNAccuracy_source_only(extractor, classifier, mnist_eval)
            source_accs.append(source_acc)
            print(f'{epoch+1}/{epochs}: source_acc: {source_acc}')
    return source_accs, extractor, classifier

def DANNTrain_target_only(mnistm_train,mnistm_eval,epochs):
    extractor = ConvolutionalExtractor().to(device)
    classifier = Classifier(3*28*28).to(device)
    class_lossf= nn.NLLLoss().to(device)
   
    optimizer = optim.SGD(
        list(extractor.parameters())+
        list(classifier.parameters()),lr=0.01,momentum=0.8)

    target_accs = []

    total_steps = epochs* len(mnistm_train)
    for epoch in range(epochs):
        start_steps = epoch * len(mnistm_train)
        
        for batch_idx, target in enumerate(mnistm_train):
            target_image, target_label = target

            p = float(batch_idx + start_steps)/ total_steps
            alpha = 2. / (1.+np.exp(-10*p))-1 
            target_image, target_label = target_image.to(device), target_label.to(device)

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01/(1.+10*p)**0.75

            # clear the grad
            optimizer.zero_grad()

            target_feature = extractor(target_image)

            # classification loss
            yp = classifier(target_feature)
            class_loss = class_lossf(yp,target_label)

            total_loss = class_loss
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            target_acc = DANNAccuracy_target_only(extractor, classifier, mnistm_eval)
            target_accs.append(target_acc)
            print(f'{epoch+1}/{epochs}: target_acc: {target_acc}')
    return target_accs, extractor, classifier
    
    
def DATrain(mnist_train, mnistm_train, mnist_eval, mnistm_eval, encoded_space_dim, epochs):
    autoencoder = DenoisingAutoencoder(encoded_space_dim).to(device)
    lossf = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.5)
    loss_arr = []
    for epoch in range(epochs):
        for batch_idx, (source, target) in enumerate(zip(mnist_train, mnistm_train)):
            source_image, source_label = source
            target_image, target_label = target

                # the source is 1 * 28 * 28, we have to preprocess it
            source_image = torch.cat((source_image, source_image, source_image),1)
            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image, target_label = target_image.to(device), target_label.to(device)
            total_image = torch.cat((source_image, target_image), 0)

            # clear the grad
            optimizer.zero_grad()
            total_image = add_noise(total_image,0.3)
            reconstructed = autoencoder(total_image)

            loss = lossf(reconstructed, total_image)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (source, target) in enumerate(zip(mnist_eval, mnistm_eval)):
                source_image, source_label = source
                target_image, target_label = target

                source_image = torch.cat((source_image, source_image, source_image),1)

                source_image, source_label = source_image.to(device), source_label.to(device)
                target_image, target_label = target_image.to(device), target_label.to(device)
                total_image = torch.cat((source_image, target_image), 0)


                reconstructed = autoencoder(total_image)
                loss = lossf(reconstructed, total_image)
                total_loss += loss
            loss_arr.append(total_loss.cpu()/len(mnist_eval))
            print(f'{epoch+1}/{epochs}: avg_loss:{total_loss/len(mnist_eval)}')
    return loss_arr, autoencoder

def DANNTrain_with_Autoencoder(autoencoder, mnist_train, mnistm_train, mnist_eval, mnistm_eval, epochs):
    extractor = LinearExtractor(16*7*7).to(device)
    classifier = Classifier(258).to(device)
    discriminator = Discriminator(258).to(device)
    for p in extractor.parameters():
        p.requires_grad = True
    for p in classifier.parameters():
        p.requires_grad = True
    for p in discriminator.parameters():
        p.requires_grad = True

    class_lossf= nn.NLLLoss().to(device)
    domain_lossf= nn.NLLLoss().to(device)
    optimizer = optim.SGD(
        list(extractor.parameters())+
        list(classifier.parameters())+
        list(discriminator.parameters()),lr=0.001,momentum=0.9)
    domain_accs =[]
    source_accs = []
    target_accs = []
    for epoch in range(epochs):
        start_steps = epoch * len(mnist_train)
        total_steps = epochs* len(mnistm_train)
        for batch_idx, (source, target) in enumerate(zip(mnist_train, mnistm_train)):
            source_image, source_label = source
            target_image, target_label = target

            p = float(batch_idx + start_steps)/ total_steps
            alpha = 2. / (1.+np.exp(-10*p))-1 

            # the source is 1 * 28 * 28, we have to preprocess it
            source_image = torch.cat((source_image, source_image, source_image),1)
            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image, target_label = target_image.to(device), target_label.to(device)
            total_image = torch.cat((source_image, target_image), 0)
            domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                    torch.ones(target_label.size()[0]).type(torch.LongTensor)),0).to(device)

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001/(1.+10*p)**0.75

            # clear the grad
            optimizer.zero_grad()
            
            total_image = autoencoder.encoder(total_image)
            source_image = autoencoder.encoder(source_image)
            total_feature = extractor(total_image)
            source_feature = extractor(source_image)

            # classification loss
            yp = classifier(source_feature)
            class_loss = class_lossf(yp,source_label)

            # domain discriminate loss
            domain_yp = discriminator(total_feature,alpha)
            domain_loss = domain_lossf(domain_yp,domain_label)

            total_loss = class_loss+domain_loss
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            source_acc, target_acc, domain_acc = DANNAccuracy_with_DA(autoencoder,extractor, classifier, discriminator, mnist_eval, mnistm_eval)
            domain_accs.append(domain_acc)
            source_accs.append(source_acc)
            target_accs.append(target_acc)
            print(f'{epoch+1}/{epochs}: source_acc: {source_acc},target_acc: {target_acc}, domain_acc: {domain_acc}')
    return source_accs, target_accs, domain_accs, extractor, classifier, discriminator
