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
from ModelOnePointTwo import *

import warnings


class customTensorData():
    """
	Custom class to load the data and transform it into tensors that are meant to be used with the DataLoader function of pytorch. 
    """
    def __init__(self,data_path,which='Train', transform=None, target_transform=None,just_one=False,column=None,tacke_offsets = False):

        x_train = np.load(data_path + '/x_train.npy')
        y_train = np.load(data_path + '/y_train.npy')
        x_test = np.load(data_path + '/x_test.npy')
        y_test = np.load(data_path + '/y_test.npy')

        #try taking the odd values of zenith
        if tacke_offsets == True:
            y_train = y_train[x_train[:,21]>10]
            x_train = x_train[x_train[:,21]>10]

            y_test = y_test[x_test[:,21]>10]
            x_test = x_test[x_test[:,21]>10]
        
        x = np.concatenate((x_train,x_test))
        y = np.concatenate((y_train,y_test))

        self.transform = transform
        self.target_transform = target_transform
        if which == 'Train':
            self.images = x_train
            self.labels = y_train
        elif which == 'Test':
            self.images = x_test
            self.labels = y_test
        elif which == 'all':
            self.images = x
            self.labels = y

        self.dates = self.images[:,25]

        self.images = np.delete(self.images,[15,16,17,18,19,22,23,24,25],axis=1) # the files contain lambda, chla, nap, cdom of first run and date.
        
        if just_one == True:
            self.images = self.images[~np.isnan(self.labels[:,column])]
            self.labels = self.labels[~np.isnan(self.labels[:,column])]


        
        
        self.labels_stds = np.nanstd(self.labels,axis=0)
        self.images_stds = np.nanstd(self.images,axis=0)
        self.labels_means = np.nanmean(self.labels,axis=0)
        self.images_means = np.nanmean(self.images,axis=0)

        self.labels_normalization_means = np.nanmean(y_train,axis=0)
        self.labels_normalization_stds = np.nanstd(y_train,axis=0)

            
        self.images_columns = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555','Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','zenith','PAR']
        self.info = '''self.dates is the number of days since the first of january of 2000 of self.images'''

        self.normalization_means = np.nanmean(np.delete(x_train,[15,16,17,18,19,22,23,24,25],axis=1),axis=0) #using only the train data for the normalization
        self.normalization_stds = np.nanstd(np.delete(x_train,[15,16,17,18,19,22,23,24,25],axis=1),axis=0)

        #self.images = (self.images - self.normalization_means)/self.normalization_stds
        #self.labels = (self.labels - self.labels_normalization_means)/self.labels_normalization_stds

        self.data_path = data_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Now that im transforming the data in tensors, I'm going to loos the names, so the order is important
        col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17
        RRS412,RRS442,RRS490,RRS510,RRS555,Edif412,Edif442,Edif490,Edif510,Edif555,Edir412,Edir442,
        Edir490,Edir510,Edir555,zenith,PAR

        and for the labels,
        col1,col2,col3,col4,col5,col6,col7,col8,col9
        chla,kd412,kd442,kd490,kd510,kd555,bbp442,bbp490,bbp555
        """
        
        label = torch.tensor(self.labels[idx])

        image = torch.tensor(self.images[idx])
          
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




init_time = time.time()
data_path = '/Users/carlos/Documents/surface_data_analisis/npy_data'

train_data = customTensorData(data_path,which='Train',tacke_offsets = False)
labels_stds = train_data.labels_stds
test_data = customTensorData(data_path,which = 'Test',tacke_offsets = False)

all_data = customTensorData(data_path,which='all',tacke_offsets = False)


y_data_no = all_data.labels[all_data.images[:,15]>10]
dates_no = all_data.dates[all_data.images[:,15]>10]
x_data_no = all_data.images[all_data.images[:,15]>10]


print(y_data_no.shape,dates_no.shape,x_data_no.shape)

images_means = train_data.normalization_means
images_stds = train_data.normalization_stds




def final_x(x,y,means,stds):

    """
    x
        col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17
        RRS412,RRS442,RRS490,RRS510,RRS555,Edif412,Edif442,Edif490,Edif510,Edif555,Edir412,Edir442,
        Edir490,Edir510,Edir555,zenith,PAR

    y
        col1,col2,col3
        chla,nap,cdom

    output shape:
        col1,col2,col3,col4,col5,col6,col7,col8,col9
        chla,kd412,kd442,kd490,kd510,kd555,bbp442,bbp490,bbp555
    
    """
    x = (x*stds)+means
    lambdas = [412.5,442.5,490.,510.,555.]
    initial_perturbation_factors = {
            'mul_factor_a_ph':1,
            'mul_tangent_b_ph':1,
            'mul_intercept_b_ph':1,
            'mul_factor_a_cdom':1,
            'mul_factor_s_cdom':1,
            'mul_factor_q_a':1,
            'mul_factor_q_b':1,
            'mul_factor_theta_min':1,
            'mul_factor_theta_o':1,
            'mul_factor_beta':1,
            'mul_factor_sigma':1,
            'mul_factor_backscattering_ph':1,
            'mul_factor_backscattering_nap':1,
        }


    x_result = torch.empty((y.size()[0],9))

    x_result[:,0] = y[:,0] 

    for i in range(5):
        
        x_result[:,1+i] =  kd(9.,x[:,5+i],x[:,10+i],lambdas[i],x[:,15],x[:,16],y[:,0],y[:,1],y[:,2],initial_perturbation_factors)
        
        if i == 0:
            x_result[:,6] = bbp(x[:,5+i],x[:,10+i],lambdas[i],x[:,15],x[:,16],y[:,0],y[:,1],y[:,2],initial_perturbation_factors) #input is Edif_555,Edir_555,555,zenit,PAR,chla,nap,cdom,perturbation_factors
        if i == 2:
            x_result[:,7] = bbp(x[:,5+i],x[:,10+i],lambdas[i],x[:,15],x[:,16],y[:,0],y[:,1],y[:,2],initial_perturbation_factors)
        if i == 3:
            x_result[:,8] = bbp(x[:,5+i],x[:,10+i],lambdas[i],x[:,15],x[:,16],y[:,0],y[:,1],y[:,2],initial_perturbation_factors)

    return x_result

class NNConvolutionalModel(nn.Module):

    def __init__(self,shape_layers = None):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 5, stride = 1, padding = 1)
        self.conv2 = nn.Conv1d(3, 5, 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv1d(3, 5, 5, stride = 1, padding = 1)
        self.ReLU = nn.ReLU()
        self.Pool = nn.MaxPool1d(2)
        self.Linear = nn.Linear(21,100)
        self.Linear2 = nn.Linear(100,3)

    def forward(self,image):
        x = torch.reshape(image,(image.size()[0],1,16))
        x = self.conv1(x)
        x = self.ReLU(x)
        #print(x.size())
        #x = self.conv2(x)
        #x = self.ReLU(x)
        #x = self.conv3(image)
        #x = self.ReLU(x)
        x = self.Pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.Linear(x)
        x = self.ReLU(x)
        x = self.Linear2(x)
        x = self.ReLU(x)
        ###x = final_x(image,x)
        return x




class NNFullyConectedModel(nn.Module):
    """
	Neural Network model for the inversion problem.
    """
    def __init__(self,shape_layers = [(17,50),(50,9)],dropout = 0.2):
        
        super().__init__()
        self.number_layers = len(shape_layers)
        self.Layers_list = []
        for i in range(self.number_layers):
            self.Layers_list.append(nn.Linear(*shape_layers[i]))
            self.Layers_list.append(nn.ReLU())  
        self.Dropout = nn.Dropout(dropout)
        self.Linear_layers = nn.Sequential(*self.Layers_list)

    def forward(self,image):
        #x = self.Dropout(image)

        x = self.Linear_layers(image)
        #x = final_x(image,x,means=images_means,stds=images_stds)

        return x

class NNFullyConectedModel2(nn.Module):
    """
	Neural Network model for the inversion problem.
    """
    def __init__(self,shape_layers = [[(17,50),(50,13)],[(13+17,50),(50,9)]],dropout = 0.2):
        
        super().__init__()
        self.number_layers = len(shape_layers)
        self.Layers_list1 = []
        for i in range(self.number_layers):
            self.Layers_list1.append(nn.Linear(*shape_layers[0][i]))
            self.Layers_list1.append(nn.ReLU())

        self.Layers_list2 = []
        for i in range(self.number_layers):
            self.Layers_list2.append(nn.Linear(*shape_layers[1][i]))
            self.Layers_list2.append(nn.ReLU())
            
        self.Dropout = nn.Dropout(dropout)
        self.Linear_layers1 = nn.Sequential(*self.Layers_list1)
        self.Linear_layers2 = nn.Sequential(*self.Layers_list2)
    def forward(self,image):
        #x = self.Dropout(image)
        x = self.Linear_layers1(image)
        x = torch.cat((x,image),1)
        x = self.Linear_layers2(x)
        #x = final_x(image,x,means=images_means,stds=images_stds)

        return x

class AutoencoderNNFullyConectedModel(nn.Module):
    """
	Neural Network model for the inversion problem.
    """
    def __init__(self,shape_layers = [(16,50),(50,9),(9,50),(50,16)],dropout=0.2):
        

        super().__init__()
        self.number_layers = len(shape_layers)
        self.Linear_encoder_list = []
        self.Linear_decoder_list = []
        for i in range(int(self.number_layers/2)):
            self.Linear_encoder_list.append(nn.Linear(*shape_layers[i]))
            self.Linear_encoder_list.append(nn.ReLU())
        for i in range(int(self.number_layers/2),self.number_layers):
            self.Linear_decoder_list.append(nn.Linear(*shape_layers[i]))
            self.Linear_decoder_list.append(nn.ReLU())
        

        self.Linear_encoder = nn.Sequential(*self.Linear_encoder_list)
        self.Linear_decoder = nn.Sequential(*self.Linear_decoder_list)
        self.Dropout = nn.Dropout(dropout)

    def forward(self,image):
        x = self.Dropout(image)
        code = self.Linear_encoder(x)
        decode = self.Linear_decoder(code)
        return decode


def train_loop(dataloader, model, loss_fn, optimizer, stds, batch_size,decoder=False):

    size = len(dataloader.dataset)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    model.train()
    train_loss = []
    for batch, (X, y_nan) in enumerate(dataloader):
        # Compute prediction and loss
        if decoder == True:
            y = torch.empty((y_nan.size()[0],9+22))
            y[:,:9] = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
            y[:,9:] = X
            pred = torch.empty(y_nan.size()[0],9+22)
            pred_code,pred_decode = model(X.float())
            pred_code = torch.masked_fill(pred_code,torch.isnan(y_nan),0)
            pred[:,:9] = pred_code
            pred[:,9:] = pred_decode
        else:
        
            y = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
            pred = model(X.float())
            pred = torch.masked_fill(pred,torch.isnan(y_nan),0)

        loss = loss_fn(pred, y, stds,y_nan)/y_nan.size()[0]

        # Backpropagation
        loss.backward()
        optimizer.step()
        #scheduler.step(loss)
        train_loss.append(float(loss.clone().detach().numpy()))
        optimizer.zero_grad()
    return np.array(train_loss)

        #if batch  %  25 == 0:
        #    loss, current = loss.item(), (batch + 1) * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, stds,batch_size,decoder=False):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for epoch,(X, y_nan) in enumerate(dataloader):
            if decoder == True:
                y = torch.empty(y_nan.size()[0],9+16)
                y[:,:9] = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
                y[:,9:] = X
                pred = torch.empty(y_nan.size()[0],9+16)
                pred_code,pred_decode = model(X.float())
                pred_code = torch.masked_fill(pred_code,torch.isnan(y_nan),0)
                pred[:,:9] = pred_code
                pred[:,9:] = pred_decode
            else:
                y = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
                pred = model(X.float())
                pred = torch.masked_fill(pred,torch.isnan(y_nan),0)
            test_loss += loss_fn(pred, y,stds,y_nan)
    test_loss /= (num_batches*y_nan.size()[0])
    return float(test_loss.clone().detach().numpy())


def plot_output(model,just_one=False,column=None):
    #pred = (model(       ((torch.tensor(all_data.images) - all_data.normalization_means)/all_data.normalization_stds).float()     ).clone().detach()*all_data.labels_normalization_stds) + all_data.labels_normalization_means
    pred = model(torch.tensor(all_data.images).float())
    years = np.arange(2006,2013)
    dates = all_data.dates
    dates = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in dates]
    test_data_init = datetime(year=2000,month=1,day=1) + timedelta(days=test_data.dates[0])
    plot_d_data = [all_data.labels.T[i]  for i in [1,2,3,4,5,6,7,8,0]]
    plot_l_data = [pred.clone().detach().numpy().T[i] for i in [1,2,3,4,5,6,7,8,0]]
    
    lambdas = [412.5,442.5,490,510,555]
    y_labels = []
    for i in range(5): y_labels.append( '$k_{d,\lambda='+str(lambdas[i])+'}(m^{-1})$')
    for i in [1,2,4]: y_labels.append( '$b_{b,p,\lambda='+str(lambdas[i])+'}(m^{-1})$')
    y_labels.append('$clorophyl-a (mg/m^3)$')

    if just_one == True:
        plot_d_data = plot_d_data[-1]
        plot_l_data = plot_l_data[-1]
        dates = np.arange(len(plot_l_data))
        plt.scatter(dates,plot_l_data,label='learned data')
        plt.scatter(dates,plot_d_data,label='original data')
        plt.legend()
        plt.show()
        
    else:
        fig, ax = plt.subplots(9, 2,width_ratios = [2.5,1])
        end = 9
        for i in range(end):
            ax[i][0].set_ylabel(y_labels[i])
            ax[i][1].set_ylabel('Distribution')

            years_flags = [False for _ in years]
            ticks_labels = years 
            ticks = []
            year = 0
            flag = False
            for j in range(len(dates)-1):
                if ((dates[j]< datetime(year = 2006 + year,month=1,day=1) ) and (dates[j+1] >= datetime(year=2006 + year,month=1,day=1))) :
                    ticks.append(j)
                    year +=1
                if (dates[j]< test_data_init) and (dates[j+1] >= test_data_init) :
                    t_data_mark = j
            ax[i][0].scatter(np.arange(len(dates)),plot_d_data[i],marker='o',label='Filtered data',color='black',s=10)
            ax[i][0].scatter(np.arange(len(dates)),plot_l_data[i],marker='x',label='Learned data',color='seagreen',s=10,alpha=0.4)
            ax[i][0].axvline(t_data_mark,color='gray',ls='--')
            xlimits  = ax[i][0].get_xlim()
            ylimits = ax[i][0].get_ylim()
            ax[i][0].fill_between(np.arange(xlimits[0],xlimits[1]), ylimits[1],y2=ylimits[0],\
                                  where = np.arange(xlimits[0],xlimits[1]) > t_data_mark, facecolor='gray', alpha=.3,label='Test data')
            ax[i][0].set_xlim(*xlimits)
            ax[i][0].set_ylim(*ylimits)
            ax[i][0].set_xticks(ticks)
            ax[i][0].set_xticklabels(ticks_labels, rotation=0, ha='center')
            ax[i][1].hist(plot_d_data[i],bins=20,label='Filtered data',density=True,color='black',edgecolor='gray')
            ax[i][1].hist(plot_l_data[i],bins=20,label='Modeled data',alpha=0.5,density=True,color='seagreen',edgecolor='darkgreen')
            ax[i][0].legend(loc='upper left')
            ax[i][0].set_xlabel('year')
            ax[i][1].legend(loc='upper right')
            ax[i][1].set_xlabel(y_labels[i])
        plt.show()


#data = pd.DataFrame(all_data.images,columns=[*np.arange(17)])
#data['date'] = all_data.dates



#dates_ = pd.DataFrame(np.arange(1875,4748,1),columns=['date'])

#data = data.merge(dates_,how='right',on='date')

def plot_input():
    
    years = np.arange(2006,2013)
    dates = dates_['date']
    #data = all_data.images.T
    
    dates = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in dates]
    test_data_init = datetime(year=2000,month=1,day=1) + timedelta(days=test_data.dates[0])
    plot_d_data = [data[i] for i in [0,1,2,3,4]]
    for i in [5,6,7,8,9]: plot_d_data.append(data[i])
    for i in [10,11,12,13,14]: plot_d_data.append(data[i])
    for i in [15,16]: plot_d_data.append(data[i])

    lambdas = [412.5,442.5,490,510,555]
    
    y_labels = []
    for i in [0,1,2,3,4]: y_labels.append( '$R_{RS,\lambda='+str(lambdas[i])+'}(sr^{-1})$')
    for i in [0,1,2,3,4]: y_labels.append( '$E_{dif,\lambda='+str(lambdas[i])+'}(W/m^2)$')
    for i in [0,1,2,3,4]: y_labels.append( '$E_{dir,\lambda='+str(lambdas[i])+'}(W/m^2)$')
    y_labels.append('$Zenith (grad)$')
    y_labels.append('$PAR (W/m^2)$')
    
    

    
    fig, ax = plt.subplots(6, 2,width_ratios = [2.5,1],figsize=(15,9))
    for i in range(6):
        ax[i][0].set_ylabel(y_labels[i],fontsize=8)
        ax[i][1].set_ylabel('Distribution',fontsize=8)

        years_flags = [False for _ in years]
        ticks_labels = years 
        ticks = []
        year = 0
        flag = False
        for j in range(len(dates)-1):
            if ((dates[j]< datetime(year = 2006 + year,month=1,day=1) ) and (dates[j+1] >= datetime(year=2006 + year,month=1,day=1))) :
                ticks.append(j)
                year +=1
            if (dates[j]< test_data_init) and (dates[j+1] >= test_data_init) :
                t_data_mark = j
        ax[i][0].scatter(np.arange(len(dates)),plot_d_data[i],marker='o',color='gray',s=10)
        #ax[i][0].scatter(np.arange(len(dates)),plot_l_data[i],marker='x',label='Learned data',color='seagreen',s=10,alpha=0.4)
        ax[i][0].axvline(t_data_mark,color='gray',ls='--')
        xlimits  = ax[i][0].get_xlim()
        ylimits = ax[i][0].get_ylim()
        ax[i][0].fill_between(np.arange(xlimits[0],xlimits[1]), ylimits[1],y2=ylimits[0],\
                          where = np.arange(xlimits[0],xlimits[1]) > t_data_mark, facecolor='gray', alpha=.3,label='Test data')
        ax[i][0].set_xlim(*xlimits)
        ax[i][0].set_ylim(*ylimits)
        ax[i][0].set_xticks(ticks)
        ax[i][0].set_xticklabels(ticks_labels, rotation=0, ha='center',fontsize=8)
        ax[i][1].hist(plot_d_data[i],bins=20,density=True,color='gray',edgecolor='black',label=y_labels[i])
        ax[i][1].set_xticklabels(ax[i][1].get_xticklabels(),fontsize=8)
        #ax[i][1].hist(plot_l_data[i],bins=20,label='Modeled data',alpha=0.5,density=True,color='seagreen',edgecolor='darkgreen')
        ax[i][0].legend(loc='upper left')
        ax[i][0].set_xlabel('year',fontsize=8)
        ax[i][1].legend(loc='upper right')
        ax[i][1].set_xlabel(y_labels[i],fontsize=8)
        ax[i][0].set_yticklabels(ax[i][0].get_yticklabels(),fontsize=8)
        ax[i][1].set_yticklabels(ax[i][1].get_yticklabels(),fontsize=8)
    plt.show()
    fig, ax = plt.subplots(6, 2,width_ratios = [2.5,1],figsize=(15,9))
    for i in range(6):
        ax[i][0].set_ylabel(y_labels[i+6],fontsize=8)
        ax[i][1].set_ylabel('Distribution',fontsize=8)

        years_flags = [False for _ in years]
        ticks_labels = years 
        ticks = []
        year = 0
        flag = False
        for j in range(len(dates)-1):
            if ((dates[j]< datetime(year = 2006 + year,month=1,day=1) ) and (dates[j+1] >= datetime(year=2006 + year,month=1,day=1))) :
                ticks.append(j)
                year +=1
            if (dates[j]< test_data_init) and (dates[j+1] >= test_data_init) :
                t_data_mark = j
        ax[i][0].scatter(np.arange(len(dates)),plot_d_data[i+6],marker='o',color='gray',s=10)
        #ax[i][0].scatter(np.arange(len(dates)),plot_l_data[i],marker='x',label='Learned data',color='seagreen',s=10,alpha=0.4)
        ax[i][0].axvline(t_data_mark,color='gray',ls='--')
        xlimits  = ax[i][0].get_xlim()
        ylimits = ax[i][0].get_ylim()
        ax[i][0].fill_between(np.arange(xlimits[0],xlimits[1]), ylimits[1],y2=ylimits[0],\
                          where = np.arange(xlimits[0],xlimits[1]) > t_data_mark, facecolor='gray', alpha=.3,label='Test data')
        ax[i][0].set_xlim(*xlimits)
        ax[i][0].set_ylim(*ylimits)
        ax[i][0].set_xticks(ticks)
        ax[i][0].set_xticklabels(ticks_labels, rotation=0, ha='center',fontsize=8)
        ax[i][1].hist(plot_d_data[i+6],bins=20,density=True,color='gray',edgecolor='black',label=y_labels[i+6])
        ax[i][1].set_xticklabels(ax[i][1].get_xticklabels(),fontsize=8)
        #ax[i][1].hist(plot_l_data[i],bins=20,label='Modeled data',alpha=0.5,density=True,color='seagreen',edgecolor='darkgreen')
        ax[i][0].legend(loc='upper left')
        ax[i][0].set_xlabel('year',fontsize=8)
        ax[i][1].legend(loc='upper right')
        ax[i][1].set_xlabel(y_labels[i+8],fontsize=8)
        ax[i][0].set_yticklabels(ax[i][0].get_yticklabels(),fontsize=8)
        ax[i][1].set_yticklabels(ax[i][1].get_yticklabels(),fontsize=8)
    plt.show()
    fig, ax = plt.subplots(5, 2,width_ratios = [2.5,1],figsize=(15,9))
    for i in range(5):
        ax[i][0].set_ylabel(y_labels[i+12],fontsize=8)
        ax[i][1].set_ylabel('Distribution',fontsize=8)

        years_flags = [False for _ in years]
        ticks_labels = years 
        ticks = []
        year = 0
        flag = False
        for j in range(len(dates)-1):
            if ((dates[j]< datetime(year = 2006 + year,month=1,day=1) ) and (dates[j+1] >= datetime(year=2006 + year,month=1,day=1))) :
                ticks.append(j)
                year +=1
            if (dates[j]< test_data_init) and (dates[j+1] >= test_data_init) :
                t_data_mark = j
        ax[i][0].scatter(np.arange(len(dates)),plot_d_data[i+12],marker='o',color='gray',s=10)
        #ax[i][0].scatter(np.arange(len(dates)),plot_l_data[i],marker='x',label='Learned data',color='seagreen',s=10,alpha=0.4)
        ax[i][0].axvline(t_data_mark,color='gray',ls='--')
        xlimits  = ax[i][0].get_xlim()
        ylimits = ax[i][0].get_ylim()
        ax[i][0].fill_between(np.arange(xlimits[0],xlimits[1]), ylimits[1],y2=ylimits[0],\
                          where = np.arange(xlimits[0],xlimits[1]) > t_data_mark, facecolor='gray', alpha=.3,label='Test data')
        ax[i][0].set_xlim(*xlimits)
        ax[i][0].set_ylim(*ylimits)
        ax[i][0].set_xticks(ticks)
        ax[i][0].set_xticklabels(ticks_labels, rotation=0, ha='center',fontsize=8)
        ax[i][1].hist(plot_d_data[i+12],bins=20,density=True,color='gray',edgecolor='black',label=y_labels[i+12])
        ax[i][1].set_xticklabels(ax[i][1].get_xticklabels(),fontsize=8)
        #ax[i][1].hist(plot_l_data[i],bins=20,label='Modeled data',alpha=0.5,density=True,color='seagreen',edgecolor='darkgreen')
        ax[i][0].legend(loc='upper left')
        ax[i][0].set_xlabel('year',fontsize=8)
        ax[i][1].legend(loc='upper right')
        ax[i][1].set_xlabel(y_labels[i+8],fontsize=8)
        ax[i][0].set_yticklabels(ax[i][0].get_yticklabels(),fontsize=8)
        ax[i][1].set_yticklabels(ax[i][1].get_yticklabels(),fontsize=8)
    plt.show()

print('time to load the data:',time.time() - init_time)

def custom_LSELoss2(input,output,stds,nan_array):
    """
    My data has some nan on it, so this function returns the least square error loss function, taking into consideration the nan elements.
    """
    #output = output[:,0]
    #input = input[:,0]
    custom_array = ((output-input))**2
    lens = torch.tensor([len(element[~element.isnan()]) for element in nan_array])
    means_output = custom_array.sum(axis=1)/lens
    return means_output.sum()
    #return custom_array.sum()

def test_model(batch_size=40,epochs=15,learning_rates=np.logspace(-6.5,-5.5,10),shape_layers=[(22,50),(50,9)],with_comments = True,custom_model = NNFullyConectedModel,dropout=0.2,decoder = False,stds = labels_stds,plot=False):

    batch_size_ = batch_size
    time_init2 = time.time()

    if with_comments == True:
        print("############ starting the test of the model with shape {}, using {} epochs, for the learning rates {}############".format(shape_layers,epochs,learning_rates))
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  #next_iteration_image,next_iteration_label = next(iter(train_dataloader))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        
    test_loss = []
    for i,learning_rate in enumerate(learning_rates):
        
        test_loss.append([])

        NNModel = custom_model(shape_layers = shape_layers,dropout=dropout).to("cpu")

        loss_fn = Custom_LSEcorrLoss #custom_LSELoss
        optimizer = torch.optim.Adam(NNModel.parameters(), lr=learning_rate,weight_decay=1e-5)

        for epoch in range(epochs):

            train_loop(train_dataloader, NNModel, loss_fn, optimizer, stds, batch_size,decoder=decoder)
            test_loss[i].append(test_loop(test_dataloader, NNModel, loss_fn, stds,batch_size,decoder=decoder))
        if with_comments == True:
            print("#learning rate {}, loss {}#".format(learning_rate,test_loss[i][-1]))
        if plot == True:
            plt.plot(np.arange(len(test_loss[i])),test_loss[i])
            #plt.yscale('log')
            plt.show()
            
    print(batch_size,time.time() - time_init2)
    return learning_rates[np.argmin(np.array(test_loss),axis=0)[-1]],np.min(np.array(test_loss),axis=0)[-1]

def best_arquitecture(number_layers = 1,layers_sizes = None,best_loss = 1e5 ):
    best_size = 1
    for sizes in [10,20,30,40]:
        if number_layers == 1:
            shape_layers = [(22,sizes),(sizes,9)]
        elif number_layers > 1:
            shape_layers = [(22,layers_sizes[0])]
            for i in range(len(layers_sizes)-1):
                shape_layers.append((layers_sizes[i],layers_sizes[i+1]))
            shape_layers.append((layers_sizes[len(layers_sizes)-1],sizes))
            shape_layers.append((sizes,9))
        learning_rate, loss = test_model(shape_layers = shape_layers,dropout=0.2,epochs=100)
        
        if loss < best_loss:
            best_loss = loss
            best_size = sizes
        print("~~~~{} layers, shape {}, loss {}. Best last layer untill now is {} neurons in last layer, with a loss of {}".format(number_layers,shape_layers,loss,best_size,best_loss))
    return best_size,best_loss

#layers_sizes = [400,350,300,250,200,150,100,50,30]
#best_loss = 1e5
#best_number_layers = 9
#for number_layers in range(10,11):
#    print("~~~",number_layers)
#    best_size,best_loss = best_arquitecture(number_layers=number_layers,layers_sizes = layers_sizes,best_loss = 1e5)
#    layers_sizes.append(best_size)

    
    #check one liyer, 200 neurons, learning rate 2.448436746822229e-06
    #check two layers, 300, 100, learning rate 1.4677992676220705e-06
    #check two layers, 300, 200, learning rate 4.0842386526745176e-07 and 2.448436746822229e-06, and 300, 300 with learning rate 5.27499706370262e-07
    #check three layers, 300,100,50 learning rate 4.0842386526745176e-07     --------> definitly learned something
    #extremely well until now, 400,300,200,100 with learning rate 1.1364636663857242e-06
    #best with 400,300,200,100,50, learning rate 8.799225435691074e-07
    #best with 400,300,200,100,50,30,10 learning rate 5.27499706370262e-07


#learning_rate,loss = test_model(learning_rates=np.logspace(-10,-1,10),epochs = 100,custom_model = AutoencoderNNFullyConectedModel,decoder = True,stds = all_stds,shape_layers = [(22,50),(50,9),(9,50),(50,22)])
#learning_rate,loss = test_model(learning_rates=[1e-6],epochs = 500,plot=False,batch_size = 50,shape_layers = [(22,50),(50,60),(60,50),(50,22),(22,9)])
#print(learning_rate,loss)


def plot_losses(shape_layers = [(22,200),(200,9)],epochs=300,learning_rate=1e-6,batch_size=40,loss_fn = custom_LSELoss,NNModel = NNFullyConectedModel,plot_loss = True):
    """ possible NNModels: NNConvolutionalModel, NNFullyConectedModel, AutoencoderNNFullyConectedModel"""
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  #next_iteration_image,next_iteration_label = next(iter(train_dataloader))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    NNModel = NNModel(shape_layers = shape_layers).to("cpu")
    optimizer = torch.optim.Adam(NNModel.parameters(), lr=learning_rate,weight_decay=0.005)
    test_loss = []
    train_loss = []
    for epoch in range(epochs):
        train_loss.append(train_loop(train_dataloader, NNModel, loss_fn, optimizer, labels_stds, batch_size,decoder=False))
        test_loss.append(test_loop(test_dataloader, NNModel, loss_fn, labels_stds,batch_size,decoder=False))
    train_loss = np.array(train_loss)
    size_epoch = train_loss.shape[1]
    train_loss = np.reshape(train_loss,train_loss.shape[0]*train_loss.shape[1])
    print(test_loss[-1])
    if plot_loss == True:
        plt.plot(np.linspace(0,epochs-1,len(train_loss)),train_loss,'--',label='train_loss')
        plt.plot(np.arange(epochs),test_loss,label='test_loss')
        #plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Number of epochs')
        plt.legend()
        plt.show()
    return train_dataloader,test_dataloader,NNModel
  


#for learning_rate in [np.linspace(1e-7,1e-6,10)[6]]:
#    train_dataloader,test_dataloader,NNModel = plot_losses(learning_rate=learning_rate,batch_size = 600,epochs=4000,shape_layers=[(22,300),(300,50),(50,9)], NNModel =   NNFullyConectedModel,plot_loss = True)
#plot_nn_chl(NNModel,test_dataloader)

"""
new_chl = 231
first_data = train_data.images.iloc[0]['date']

print(train_data.images.iloc[0]['chla'])
data = train_data.images


train_data.images.iloc[0]['chla'] = 1

print(new_chl,train_data.images.iloc[0]['chla'])

"""

#learning_rate,loss = test_model(learning_rates=np.logspace(-10,-1,10),epochs = 10,custom_model = NNFullyConectedModel,decoder = False,stds = labels_stds,shape_layers = [(22,200),(200,100),(100,50),(50,3)],plot=False,batch_size=600)
#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,21),(21,9)],epochs=300,learning_rate=0.01,batch_size=600,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = True) 
#for i in np.arange(100):
#    print("#",i)
#    train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,17),(17,20),(20,27),(27,i),(i,9)],epochs=100,learning_rate=0.01,batch_size=600,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = False) 
#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,21),(21,9)],epochs=300,learning_rate=0.01,batch_size=600,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = True) 
#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(16,199),(199,50+19),(50+19,9)],epochs=2000,learning_rate=np.linspace(1e-5,10e-5,10)[5],batch_size=500,loss_fn = custom_LSELoss,NNModel = NNFullyConectedModel,plot_loss = True)
#try many iterations, for one layer, best is 213 neurons with loss = 3.2

#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,20+6),(20+6,9)],epochs=300,learning_rate=0.01,batch_size=1705,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = True) 
#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,21),(21,9)],epochs=300,learning_rate=0.0001668100537200059,batch_size=600,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = True) #0.83
#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,21),(21,14),(14,9)],epochs=300,learning_rate=0.0001668100537200059,batch_size=600,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = True) #0.83
#plot_output(NNModel)
#plot_input()


