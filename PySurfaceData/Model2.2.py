#!/usr/bin/env python

"""
Inversion problemas a NN.

As part of the National Institute of Oceanography, and Applied Geophysics, I'm working on an inversion problem. A detailed description can be found at
https://github.com/carlossoto362/firstModelOGS.

"""
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import pandas as pd
import os
import scipy
from scipy import stats
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import tempfile
import torch.distributed as dist
import sys

import warnings
from ModelOnePointFive import *


class NNConvolutionalModel(nn.Module):

    def __init__(self,constant = None,x_mean=None,x_std=None,y_mean=None,y_std=None,precision = torch.float32,batch_size=1):
        super().__init__()

        
        self.constant = constant
        self.x_mean = torch.tensor(x_mean).to(precision)
        self.x_std = torch.tensor(x_std).to(precision)
        self.y_mean = torch.tensor(y_mean).to(precision)
        self.y_std = torch.tensor(y_std).to(precision)
        
        self.conv1 = nn.Conv1d(1, 5, 5, stride = 1, padding = 1)
        self.conv2 = nn.Conv1d(5, 5, 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv1d(5, 5, 5, stride = 1, padding = 1)
        self.CELU = nn.CELU()
        self.ReLU = nn.ReLU()
        self.Linear1 = nn.Linear(17,20)
        self.Linear2 = nn.Linear(20,20)
        self.Linear3 = nn.Linear(20,20)
        self.Linear4 = nn.Linear(20,20)
        self.Linear5 = nn.Linear(20,3)

        self.Forward_Model = Forward_Model(learning_chla = False,num_days=batch_size, learning_perturbation_factors = True)



    def rearange_RRS(self,x):
        lambdas = torch.tensor([412.5,442.5,490.,510.,555.])
        x_ = x*self.x_std + self.x_mean
        output = torch.empty((len(x),5,5))
        output[:,:,0] = x_[:,0,5:10]
        output[:,:,1] = x_[:,0,10:15]
        output[:,:,2] = lambdas
        output[:,:,3] = x_[:,:,15]
        output[:,:,4] = x_[:,:,16]
        return output

    def forward(self,image,images_means=None,images_stds=None,labels_means=None,labels_stds=None):
        y = self.conv1(image)
        y = self.CELU(y)
        y = self.conv2(y)
        y = self.CELU(y)
        y = self.conv3(y)
        y = self.CELU(y)
        y = torch.flatten(image,1)        
        y = self.Linear1(y)
        y = self.CELU(y)
        y = self.Linear2(y)
        y = self.CELU(y)
        y = self.Linear3(y)
        y = self.CELU(y)
        y = self.Linear4(y)
        y = self.CELU(y)
        y = self.Linear5(y)
        y = self.CELU(y)
        
        y = y.unsqueeze(1)
        y_use = y.clone()
        y_use[:,:,0] = y[:,:,0] * self.y_std[0] + self.y_mean[0]


        X = self.rearange_RRS(image)
        rrs_ = self.Forward_Model(X,parameters = y_use,constant = self.constant)
        rrs_ = (rrs_ - self.x_mean[:5])/self.x_std[:5]
 
        kd_ = kd(9.,X[:,:,0],X[:,:,1],X[:,:,2],X[:,:,3],X[:,:,4],torch.exp(y_use[:,:,0]),torch.exp(y_use[:,:,1]),torch.exp(y_use[:,:,2]),self.Forward_Model.perturbation_factors,constant = self.constant)
        kd_ = (kd_  - self.y_mean[1:6])/self.y_std[1:6]

        bbp_ = bbp(X[:,:,0],X[:,:,1],X[:,:,2],X[:,:,3],X[:,:,4],torch.exp(y_use[:,:,0]),torch.exp(y_use[:,:,1]),torch.exp(y_use[:,:,2]),self.Forward_Model.perturbation_factors,constant = self.constant)[:,[1,2,4]]
        bbp_ = (bbp_ - self.y_mean[6:])/self.y_std[6:9]
        y[:,:,1:] = y_use[:,:,1:]

        return y[:,0,:],kd_,bbp_,rrs_,y_use

class composed_loss_function(nn.Module):

    def __init__(self,precision = torch.float32,my_device = 'cpu'):
        super(composed_loss_function, self).__init__()
        self.precision = precision


    def forward(self,chla,chla_pred,kd_,kd_pred,bbp_,bbp_pred,rrs_,rrs_pred,nan_array,param):

        custom_array = ((rrs_-rrs_pred)**2).mean(axis=1) + (chla - chla_pred)**2  + ((kd_-kd_pred)**2).sum(axis=1) + ((bbp_-bbp_pred)**2).sum(axis=1) 
        lens = torch.tensor([len(element[~element.isnan()])  for element in nan_array])

        means_output = custom_array/lens
        return means_output.mean().to(self.precision) + 0.001*param.mean()

def train_loop(dataloader, model, loss_fn, optimizer):

    num_batches= len(dataloader)
    total_size = len(dataloader.dataset)
    
    model.train()
    train_loss = []
    perturbation_factors_history = np.empty((14))
    chla_history = np.empty((total_size,3))
    kd_history = np.empty((total_size,5))
    bbp_history = np.empty((total_size,3))

    k = 0
    for batch, (X, Y) in enumerate(dataloader):
        Y_composed = torch.masked_fill(Y,torch.isnan(Y),0)
        chla_,kd_,bbp_,rrs_,y_use= model(X)
        kd_ = torch.masked_fill(kd_,torch.isnan(Y[:,0,1:6]),0)
        bbp_ = torch.masked_fill(bbp_,torch.isnan(Y[:,0,6:9]),0)
        chla_[:,0] = torch.masked_fill(chla_[:,0],torch.isnan(Y[:,0,0]),0)

        loss = loss_fn(chla_[:,0],Y_composed[:,0,0],kd_,Y_composed[:,0,1:6],bbp_,Y_composed[:,0,6:9],rrs_,X[:,0,:5],Y,y_use)

        perturbation_factors_history = list(iter(model.parameters()))[-1].clone().detach().numpy()
        chla_history[k:k+len(X)] = chla_.clone().detach().numpy()
        kd_history[k:k+len(X)] = kd_.clone().detach().numpy()
        bbp_history[k:k+len(X)] = bbp_.clone().detach().numpy()

        k+= len(X)

        
        loss.backward()

        optimizer.step()
        #model.state_dict()['Forward_Model.perturbation_factors'].data.clamp_(0.01,1.)

        train_loss.append(loss.clone().detach().numpy())

        #print(list(iter(model.parameters()))[-1].grad)

        optimizer.zero_grad()
        scheduler.step(loss)
    return np.mean(train_loss), chla_history, kd_history, bbp_history, perturbation_factors_history

def test_loop(t_dataloader, model, loss_fn, ):
    
    model.eval()
    with torch.no_grad():
        for  (X, Y) in t_dataloader:
            Y_composed = torch.masked_fill(Y,torch.isnan(Y),0)

            chla_,kd_,bbp_,rrs_,y_use= model(X)
            kd_ = torch.masked_fill(kd_,torch.isnan(Y[:,0,:5]),0)
            bbp_ = torch.masked_fill(bbp_,torch.isnan(Y[:,0,5:8]),0)
            chla_ = torch.masked_fill(chla_,torch.isnan(Y[:,0,8:11]),0)
            loss_ = loss_fn(chla_[:,0],Y_composed[:,0,0],kd_,Y_composed[:,0,1:6],bbp_,Y_composed[:,0,6:9],rrs_,X[:,0,:5],Y,y_use)
    return loss_.clone().detach().numpy()
    



def get_training_history(dataloader,test_dataloader,model,loss_fn,optimizer,epochs,batch_size, save = False, save_path = None):

    test_loss = []
    test_loss, train_loss, chla_history, kd_history, bbp_history, perturbation_factors_history = np.empty((epochs)), np.empty((epochs)), np.empty((epochs,data.len_data,3)),\
        np.empty((epochs,data.len_data,5)), np.empty((epochs,data.len_data,3)), np.empty((epochs,14))


    for j in range(epochs):
        train_loss_i, chla_i, kd_i, bbp_i, per_i  = train_loop(dataloader,model,loss_fn,optimizer)
        train_loss[j] = train_loss_i
        chla_history[j] = chla_i
        kd_history[j] = kd_i
        bbp_history[j] = bbp_i
        perturbation_factors_history[j] = per_i
        test_loss[j] = test_loop(test_dataloader, model, loss_fn)
        if j %100 == 0 :
            print(j,'done')
        
    if save == True:
        np.save(save_path + '/train_loss_NN.npy',train_loss)
        np.save(save_path + '/test_loss_NN.npy',test_loss)
        np.save(save_path + '/chla_history_NN.npy',chla_history)
        np.save(save_path + '/kd_history_NN.npy',kd_history)
        np.save(save_path + '/bbp_history_NN.npy',bbp_history)
        np.save(save_path + '/perturbation_factors_history_NN.npy',perturbation_factors_history)

    return train_loss,test_loss,chla_history,kd_history,bbp_history,perturbation_factors_history

data_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/npy_data'
my_device = torch.device("mps")
my_device = 'cpu'
constant = read_constants(file1='cte_lambda.csv',file2='cst.csv',my_device = my_device)
data = customTensorData(data_path=data_path,which='train',per_day = False,randomice=False,one_dimensional = True,seed = 1853)
test_data =  customTensorData(data_path=data_path,which='test',per_day = False,randomice=False,one_dimensional = True,seed = 1853)
batch_size = data.len_data
chla_mean = data.y_mean[0]
chla_std = data.y_std[0]


lr_ = 0.01
loss_fn = composed_loss_function()
model = NNConvolutionalModel(constant = constant,x_mean = data.x_mean,x_std = data.x_std, y_mean = data.y_mean, y_std = data.y_std,batch_size = batch_size)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(data, batch_size=test_data.len_data, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(),lr=lr_)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
epochs = 501
    
#train_loss,test_loss,chla_history,kd_history,bbp_history,perturbation_factors_history = get_training_history(dataloader,test_dataloader,model,loss_fn,optimizer,\
#                                                                                                            epochs,batch_size,save=True,save_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data/')
#print(lr_,test_loss[-1])

#torch.save(model.state_dict(),'/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data/model_state_dict_NN.npy')


data = customTensorData(data_path=data_path,which='all',per_day = False,randomice=False,one_dimensional = True,seed = 1853)
dataloader = DataLoader(data, batch_size=data.len_data, shuffle=False)

load_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data'
model.load_state_dict(torch.load(load_path + '/model_state_dict_NN.npy'))
model.eval()

#perturbation_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data/perturbation_factors'
#perturbation_factors = torch.tensor(np.load(perturbation_path + '/perturbation_factors_history_gaussian.npy')[-1]).to(torch.float32)
#print(model.state_dict()['Forward_Model.perturbation_factors'])

#list(model.parameters())[-1] = perturbation_factors
#print(model.state_dict()['Forward_Model.perturbation_factors'])

train_loss = np.load(load_path + '/train_loss_NN.npy')
test_loss = np.load(load_path + '/test_loss_NN.npy')
perturbation_factors_history = np.load(load_path + '/perturbation_factors_history_NN.npy')

plot_tracked_parameters(perturbation_factors_history = perturbation_factors_history, test_loss = test_loss, train_loss = train_loss, save_file_name = '/perturbation_factors_NN.pdf',save=True,color_palet=1,side = 'left',with_loss =False)

X,Y = next(iter(dataloader))
chla_,kd_,bbp_,rrs_,y_use= model(X)

chla_ = chla_.clone().detach().numpy()
chla_[:,0] = chla_[:,0]* chla_std + chla_mean

kd_ = kd_.clone().detach().numpy() * data.y_std[1:6] + data.y_mean[1:6]
bbp_ = bbp_.clone().detach().numpy() * data.y_std[6:] + data.y_mean[6:]
rrs_ = rrs_.clone().detach().numpy() * data.x_std[:5] + data.x_mean[:5]

#plt.plot(data.dates,chla_[:,0],label='prediction')
#plt.plot(data.dates,data.y_normilized[:,0],label='data')
#plt.legend()
#plt.show()

#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_NN_NNparam/X_hat.npy',chla_)
#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_NN_NNparam/RRS_hat.npy',rrs_)
#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_NN_NNparam/kd_hat.npy',kd_)
#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_NN_NNparam/bbp_hat.npy',bbp_)


#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_NN_NNparam/dates.npy',data.dates)







    
