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

def DANNAccuracy(extractor, classifier, discriminator, mnist_gen, mnistm_gen):
    s_cor = 0
    t_cor = 0
    domain_cor = 0
    for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
        p = float(batch_idx)/len(mnist_gen)
        alpha = 2. / (1.+np.exp(-10*p))-1 
        source_image, source_label = source
        target_image, target_label = target
            
        source_image = torch.cat((source_image, source_image, source_image),1)
        
        source_image, source_label = source_image.to(device), source_label.to(device)
        target_image, target_label = target_image.to(device), target_label.to(device)
        total_image = torch.cat((source_image, target_image), 0)
        
        domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                torch.ones(target_label.size()[0]).type(torch.LongTensor)),0).to(device)
                
        total_feature = extractor(total_image)
        source_feature = extractor(source_image)
        target_feature = extractor(target_image)
        # classification loss
        yp_s = classifier(source_feature)
        #class_loss = lossf(yp,source_label)
        yp_s = yp_s.data.max(1,keepdim=True)[1]
        #print(yp)
        s_cor += yp_s.eq(source_label.data.view_as(yp_s)).cpu().sum()
        #print(yp.eq(source_label.data.view_as(yp)).cpu().sum())
        
        yp_t = classifier(target_feature)
        #class_loss = lossf(yp,source_label)
        yp_t = yp_t.data.max(1,keepdim=True)[1]
        #print(yp)
        t_cor += yp_t.eq(target_label.data.view_as(yp_t)).cpu().sum()
        
        # domain discriminate loss
        domain_yp = discriminator(total_feature,alpha)
        #domain_loss = lossf(domain_yp,domain_label)
        domain_yp = domain_yp.data.max(1,keepdim=True)[1]
        domain_cor += domain_yp.eq(domain_label.data.view_as(domain_yp)).cpu().sum()
        #print(domain_yp.eq(domain_label.data.view_as(domain_yp)).cpu().sum())
            
        #domain_loss_avg.update(domain_loss)
        #classifier_loss_avg.update(class_loss)
            
    #domain_losses.append(-domain_loss_avg.avg.cpu().item())
    #classifier_losses.append(-classifier_loss_avg.avg.cpu().item())
    domain_acc = domain_cor.item()/(len(mnist_gen.dataset)+len(mnistm_gen.dataset))
    s_acc = s_cor.item()/len(mnist_gen.dataset)
    t_acc = t_cor.item()/len(mnistm_gen.dataset)
    return s_acc, t_acc, domain_acc
def DANNAccuracy_with_DA(autoencoder,extractor, classifier, discriminator, mnist_gen, mnistm_gen):
    s_cor = 0
    t_cor = 0
    domain_cor = 0
    for batch_idx, (source, target) in enumerate(zip(mnist_gen, mnistm_gen)):
        p = float(batch_idx)/len(mnist_gen)
        alpha = 2. / (1.+np.exp(-10*p))-1 
        source_image, source_label = source
        target_image, target_label = target
            
        source_image = torch.cat((source_image, source_image, source_image),1)
        
        source_image, source_label = source_image.to(device), source_label.to(device)
        target_image, target_label = target_image.to(device), target_label.to(device)
        total_image = torch.cat((source_image, target_image), 0)
        
        domain_label = torch.cat((torch.zeros(source_label.size()[0]).type(torch.LongTensor),
                                torch.ones(target_label.size()[0]).type(torch.LongTensor)),0).to(device)
        
        total_image = autoencoder.encoder(total_image)
        source_image = autoencoder.encoder(source_image)
        target_image = autoencoder.encoder(target_image)
        
        total_feature = extractor(total_image)
        source_feature = extractor(source_image)
        target_feature = extractor(target_image)
        # classification loss
        yp_s = classifier(source_feature)
        #class_loss = lossf(yp,source_label)
        yp_s = yp_s.data.max(1,keepdim=True)[1]
        #print(yp)
        s_cor += yp_s.eq(source_label.data.view_as(yp_s)).cpu().sum()
        #print(yp.eq(source_label.data.view_as(yp)).cpu().sum())
        
        yp_t = classifier(target_feature)
        #class_loss = lossf(yp,source_label)
        yp_t = yp_t.data.max(1,keepdim=True)[1]
        #print(yp)
        t_cor += yp_t.eq(target_label.data.view_as(yp_t)).cpu().sum()
        
        # domain discriminate loss
        domain_yp = discriminator(total_feature,alpha)
        #domain_loss = lossf(domain_yp,domain_label)
        domain_yp = domain_yp.data.max(1,keepdim=True)[1]
        domain_cor += domain_yp.eq(domain_label.data.view_as(domain_yp)).cpu().sum()
        #print(domain_yp.eq(domain_label.data.view_as(domain_yp)).cpu().sum())
            
        #domain_loss_avg.update(domain_loss)
        #classifier_loss_avg.update(class_loss)
            
    #domain_losses.append(-domain_loss_avg.avg.cpu().item())
    #classifier_losses.append(-classifier_loss_avg.avg.cpu().item())
    domain_acc = domain_cor.item()/(len(mnist_gen.dataset)+len(mnistm_gen.dataset))
    s_acc = s_cor.item()/len(mnist_gen.dataset)
    t_acc = t_cor.item()/len(mnistm_gen.dataset)
    return s_acc, t_acc, domain_acc