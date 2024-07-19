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
    def __init__(self,data_path='./npy_data',transform=None,target_transform=None,train_percentage = 0.9,from_where = 'left',randomice=False,without_off_values=True,specific_columns=None,which='train',seed=None,identity=False):
        if without_off_values == True:
            add = '_no_off'
        else:
            add = ''
        self.x_data = np.load(data_path + '/x_data' + add + '.npy')

        if identity == True:
            self.y_data = self.x_data[:,:5]
            self.y_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555']
        else:
            self.y_data = np.load(data_path + '/y_data' + add + '.npy')
            self.y_column_names = ['chla','kd_412','kd_442','kd_490','kd_510','kd_555','bbp_442','bbp_490','bbp_555']
        self.dates = np.load(data_path + '/dates' + add + '.npy')

        self.x_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555','Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','zenith','PAR']


        if specific_columns != None:
            self.x_data = self.x_data[specific_columns]
            self.x_column_names = self.x_column_names[specific_columns]

        self.len_data = len(self.dates)
        self.indexes = np.arange(self.len_data)
        if randomice == True:
            if seed !=None:
                np.random.seed(seed)
            np.random.shuffle(self.indexes)

        if from_where == 'right':
            self.indexes = np.flip(self.indexes)

        self.train_indexes = self.indexes[:int(self.len_data * train_percentage)]
        self.test_indexes = self.indexes[int(self.len_data * train_percentage):]


        self.x_normalization_means = np.nanmean(self.x_data[self.train_indexes],axis=0)
        self.x_normalization_stds = np.nanstd(self.x_data[self.train_indexes],axis=0)
        self.y_normalization_means = np.nanmean(self.y_data[self.train_indexes],axis=0)
        self.y_normalization_stds = np.nanstd(self.y_data[self.train_indexes],axis=0)

        self.x_data_normalize = (self.x_data - self.x_normalization_means)/self.x_normalization_stds
        self.y_data_normalize = (self.y_data - self.y_normalization_means)/self.y_normalization_stds

        self.which = which
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.my_indexes = self.indexes
        if self.which.lower().strip() == 'train' :
            self.my_indexes = self.train_indexes
        elif self.which.lower().strip() == 'test':
            self.my_indexse = self.test_indexes
        self.identity = identity


    def __len__(self):

            return len(self.my_indexes)

    def __getitem__(self, idx):
        

        
        label = torch.tensor(self.y_data_normalize[self.my_indexes][idx])

        image = torch.tensor(self.x_data_normalize[self.my_indexes][idx])
          
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def final_x(x,y,means,stds,labels_means,labels_stds,perturbation_factors):

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
    y[y<0] = 0

    lambdas = [412.5,442.5,490.,510.,555.]
    initial_perturbation_factors = perturbation_factors


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

    return (x_result - torch.tensor(labels_means))/torch.tensor(labels_stds)


def final2_x(x,y,means,stds,labels_means,labels_stds,perturbation_factors):

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
    
    perturbation_factors_ = {
            'mul_factor_a_ph':torch.tensor(1),
            'mul_tangent_b_ph':torch.tensor(1),
            'mul_intercept_b_ph':torch.tensor(1),
            'mul_factor_a_cdom':torch.tensor(1),
            'mul_factor_s_cdom':torch.tensor(1),
            'mul_factor_q_a':torch.tensor(1),
            'mul_factor_q_b':torch.tensor(1),
            'mul_factor_theta_min':torch.tensor(1),
            'mul_factor_theta_o':torch.tensor(1),
            'mul_factor_beta':torch.tensor(1),
            'mul_factor_sigma':torch.tensor(1),
            'mul_factor_backscattering_ph':torch.tensor(1),
            'mul_factor_backscattering_nap':torch.tensor(1),
    }

    """
    
    x = (x*stds)+means
    y_use = y.clone()
    y_use[y_use<0]=0
    

    lambdas = [412.5,442.5,490.,510.,555.]
    
    Rrs = torch.empty(x.size()[0],5)

    for i in range(5):
        Rrs[:,i] = Rrs_MODEL(x[:,5+i],x[:,10+i],lambdas[i],\
                            x[:,15],x[:,16],y_use[:,0],y_use[:,1],y_use[:,2],perturbation_factors)
    return (Rrs  - torch.tensor(labels_means))/torch.tensor(labels_stds)

class NNConvolutionalModel(nn.Module):

    def __init__(self,shape_layers = None):
        super().__init__()

        
        
        self.conv1 = nn.Conv1d(1, 2, 4, stride = 1, padding = 1)
        self.conv2 = nn.Conv1d(5, 2, 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv1d(5, 5, 5, stride = 1, padding = 1)
        self.CELU = nn.CELU()

        self.Pool = nn.MaxPool1d(2)
        self.Linear1 = nn.Linear(32,20)
        self.Linear2 = nn.Linear(20,20)
        self.Linear3 = nn.Linear(20,20)
        self.Linear4 = nn.Linear(20,20)
        self.Linear5 = nn.Linear(20,9)
        self.Linear6 = nn.Linear(20,9)
        #self.Linear2 = nn.Linear(100,3)

    def forward(self,image,images_means=None,images_stds=None,labels_means=None,labels_stds=None):
        x = torch.reshape(image,(image.size()[0],1,17))
        x = self.conv1(x)
        x = self.CELU(x)
        #x = self.conv2(x)
        #x = self.CELU(x)
        #x = self.conv3(x)
        #x = self.CELU(x)
        #x = self.Pool(x)
        #print(x.size())
        x = torch.flatten(x,start_dim=1)
        x = self.Linear1(x)
        x = self.CELU(x)
        x = self.Linear2(x)
        x = self.CELU(x)
        x = self.Linear3(x)
        x = self.CELU(x)
        x = self.Linear4(x)
        x = self.CELU(x)
        x = self.Linear5(x)
        x = self.CELU(x)
        #x = self.Linear6(x)
        #x = self.CELU(x)
        ###x = final_x(image,x,means=images_means,stds=images_stds,labels_means=labels_means,labels_stds=labels_stds,perturbation_factors=self.perturbation_factors)
        return x




class NNFullyConectedModel(nn.Module):
    """
	Neural Network model for the inversion problem.
    """
    def __init__(self,shape_layers = [(17,50),(50,9)],dropout = 0.2):
        
        super().__init__()
        self.mul_factors = nn.Parameter(torch.ones(13, dtype=torch.float32), requires_grad=True)
        
        self.number_layers = len(shape_layers)
        self.Layers_list = []
        for i in range(self.number_layers):
            self.Layers_list.append(nn.Linear(*shape_layers[i]))
            self.Layers_list.append(nn.CELU())  
        self.Dropout = nn.Dropout(dropout)
        self.Linear_layers = nn.Sequential(*self.Layers_list)

    def forward(self,image,images_means=None,images_stds=None,labels_means=None,labels_stds=None):
        #x = self.Dropout(image)

        x = self.Linear_layers(image)
        #x = final_x(image,x,means=images_means,stds=images_stds,labels_means=labels_means,labels_stds=labels_stds,perturbation_factors=self.perturbation_factors)

        return x


class NNFullyConectedModel2(nn.Module):
    """
	Neural Network model for the inversion problem.
    """
    def __init__(self,shape_layers = None,dropout = 0.2):
        
        super().__init__()

        Model1 = NNFullyConectedModel(shape_layers = [(17,20),(20,20),(20,20),(20,20),(20,20),(20,9)]).to("cpu")
        Model1.load_state_dict(torch.load('nngraphs/FullyConnected4layersOf20.pt'))


        
        Model1.eval()
        self.Model1 = Model1
        self.chl_mean = 0.44296421
        self.chl_std = 0.55154612   #this are the values with the train data that the original value was trained. 

        self.mul_factors = nn.Parameter(torch.ones(13, dtype=torch.float32), requires_grad=True)

        self.chla_epsilon = nn.Parameter(torch.zeros(1,dtype=torch.float32),requires_grad=True)

        self.perturbation_factors = {
            'mul_factor_a_ph':self.mul_factors[0],
            'mul_tangent_b_ph':self.mul_factors[1],
            'mul_intercept_b_ph':self.mul_factors[2],
            'mul_factor_a_cdom':self.mul_factors[3],
            'mul_factor_s_cdom':self.mul_factors[4],
            'mul_factor_q_a':self.mul_factors[5],
            'mul_factor_q_b':self.mul_factors[6],
            'mul_factor_theta_min':self.mul_factors[7],
            'mul_factor_theta_o':self.mul_factors[8],
            'mul_factor_beta':self.mul_factors[9],
            'mul_factor_sigma':self.mul_factors[10],
            'mul_factor_backscattering_ph':self.mul_factors[11],
            'mul_factor_backscattering_nap':self.mul_factors[12],
        }

        self.Layer1 = nn.Sequential(
            nn.Linear(9,20),
            nn.CELU(),
            nn.Linear(20,20),
            nn.CELU(),
            nn.Linear(20,2),
            nn.CELU()
        )
        
    def forward(self,image,images_means=None,images_stds=None,labels_means=None,labels_stds=None):

        x = self.Model1(image)
        chla = (x[:,0]*self.chl_std) + self.chl_mean + self.chla_epsilon
        
        x = self.Layer1(x)

        new_x = torch.empty((x.size()[0],3))
        new_x[:,0] = chla
        new_x[:,1:] = x
        
        x = final2_x(image,new_x,means=images_means,stds=images_stds,labels_means=labels_means,labels_stds=labels_stds,perturbation_factors=self.perturbation_factors)
        
        return x,new_x.clone().detach()



def train_loop(dataloader, model, loss_fn, optimizer, batch_size,decoder=False,images_means=None,images_stds=None,labels_means=None,labels_stds=None,identity=False):

    size = len(dataloader.dataset)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    model.train()
    train_loss = []
    for batch, (X, y_nan) in enumerate(dataloader):
        # Compute prediction and loss
        y = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
        if identity == False:
            pred = model(X.float(),images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds)
        else:
            pred = model(X.float(),images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds)[0]
        pred = torch.masked_fill(pred,torch.isnan(y_nan),0)
        loss = loss_fn(pred, y,y_nan)/y_nan.size()[0]
        #print(pred,loss)
        # Backpropagation
        loss.backward()
        optimizer.step()
        if identity == True:
            model.state_dict()['chla_epsilon'].data.clamp_(-0.22,0.22)
            model.state_dict()['mul_factors'].data.clamp_(0.5,1.5)
        #scheduler.step(loss)
        train_loss.append(float(loss.clone().detach().numpy()))
        optimizer.zero_grad()
    return np.array(train_loss)

        #if batch  %  25 == 0:
        #    loss, current = loss.item(), (batch + 1) * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn,batch_size,decoder=False,images_means=None,images_stds=None,labels_means=None,labels_stds=None,identity=False):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for epoch,(X, y_nan) in enumerate(dataloader):

            y = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
            if identity == False:
                pred = model(X.float(),images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds)
            else:
                pred = model(X.float(),images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds)[0]
            pred = torch.masked_fill(pred,torch.isnan(y_nan),0)
            test_loss += loss_fn(pred, y,y_nan)
    test_loss /= (num_batches*y_nan.size()[0])
    return float(test_loss.clone().detach().numpy())


def plot_output(model,data,just_one=False,column=None,images_means=None,images_stds=None,labels_means=None,labels_stds=None,identity=False,indexes=None):

    years = np.arange(2006,2013)
    dates = data.dates
    dates = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in dates]
    test_data_init = datetime(year=2000,month=1,day=1) + timedelta(days=test_data.dates[0])

    if identity == False:
        pred = model(torch.tensor(data.x_data_normalize).float(),images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds)
        if type(labels_means) == type(None):
            pass
        else:
            pred = (pred * torch.tensor(labels_stds)) + torch.tensor(labels_means)
        
        if type(indexes) == type(None):
            y_data = data.y_data
        else:
            pred = pred[indexes]
            y_data = data.y_data[indexes]
            dates = np.array(dates)[indexes]
            
        
        plot_d_data = [y_data.T[i]  for i in [1,2,3,4,5,6,7,8,0]]
        plot_l_data = [pred.clone().detach().numpy().T[i] for i in [1,2,3,4,5,6,7,8,0]]
    
        lambdas = [412.5,442.5,490.,510.,555.]
        y_labels = []
        for i in range(5): y_labels.append( '$k_{d,\lambda='+str(lambdas[i])+'}(m^{-1})$')
        for i in [1,2,4]: y_labels.append( '$b_{b,p,\lambda='+str(lambdas[i])+'}(m^{-1})$')
        y_labels.append('$clorophyl-a (mg/m^3)$')

        if just_one == True:
            plot_d_data = plot_d_data[-1]
            plot_l_data = plot_l_data[-1]
            plt.scatter(dates,plot_l_data,label='learned data')
            plt.scatter(dates,plot_d_data,label='original data')
            plt.legend()
            plt.show()
        
        else:
            def plot_some_output(start,end):

                fig, ax = plt.subplots(end - start, 2,width_ratios = [2.5,1],figsize=(18, 9))
                for i in range(start,end):

                    ks = stats.kstest(plot_d_data[i][~np.isnan(plot_d_data[i])],plot_l_data[i][~np.isnan(plot_d_data[i])])
                    corr = stats.pearsonr(plot_d_data[i][~np.isnan(plot_d_data[i])],plot_l_data[i][~np.isnan(plot_d_data[i])])
                    rmse = np.sqrt(np.mean((plot_d_data[i][~np.isnan(plot_d_data[i])]-plot_l_data[i][~np.isnan(plot_d_data[i])])**2))
                    ax[i-start][0].set_ylabel(y_labels[i])
                    ax[i-start][1].set_ylabel('Distribution')

                    ax[i-start][0].scatter(dates,plot_d_data[i],marker='o',label='Filtered data',color='black',s=10)
                    ax[i-start][0].scatter(dates,plot_l_data[i],marker='x',label='Learned data',color='seagreen',s=10,alpha=0.4)
                    ax[i-start][1].hist(plot_d_data[i],bins=20,label='Filtered data',density=True,color='black',edgecolor='gray')
                    ax[i-start][1].hist(plot_l_data[i][~np.isnan(plot_d_data[i])],bins=20,label='Modeled data',alpha=0.5,density=True,color='seagreen',edgecolor='darkgreen')
                    ax[i-start][0].legend(loc='upper left')
                    ax[i-start][0].set_xlabel('year')
                    ax[i-start][1].legend(loc='upper right')
                    ax[i-start][1].set_xlabel(y_labels[i])
                    #props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
                    my_text = "KS Test: {:.3f}, p-value: {:.3E}\nRMSE: {:.3E}\nCorr: {:.3f}".format(ks.statistic,ks.pvalue,rmse,corr.statistic)
                    if (start==0) and (i==1):
                        ax[i-start][1].text(0.26, 0.83, my_text, transform=ax[i-start][1].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)
                    else:
                        ax[i-start][1].text(0.55, 0.5, my_text, transform=ax[i-start][1].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)
                    fig.tight_layout()
                plt.show()
            
            plot_some_output(0,5)
            plot_some_output(5,9)

            #noice distribution of chla
            chla_d = plot_d_data[-1][~np.isnan(plot_d_data[-1])]
            chla_l = plot_l_data[-1][~np.isnan(plot_d_data[-1])]
            error = chla_d - chla_l    
    else:

        pred,chls = model(torch.tensor(data.x_data_normalize).float(),images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds)

        if type(labels_means) == type(None):
            pass
        else:
            pred = (pred * torch.tensor(labels_stds)) + torch.tensor(labels_means)
            
        if type(indexes) == type(None):
            y_data = data.y_data
        else:
            pred = pred[indexes]
            y_data = data.y_data[indexes]
            dates = np.array(dates)[indexes]
            chls = chls[indexes]
        
        fig, ax = plt.subplots(3, 1,figsize=(15,9))
        ax[0].scatter(dates,chls[:,0],marker='o',label='Chlorophyll-a',color='darkgreen',s=10)
        ax[0].set_xlabel('year')
        ax[0].set_ylabel('Chlorophyll-a $mg/m^3$')
        ax[0].legend()
        ax[1].scatter(dates,chls[:,1],marker='o',label='NAP',color='lightblue',s=10)
        ax[1].set_ylabel('Non Algal Particles $mg/m^3$')
        ax[1].set_xlabel('year')
        ax[1].legend()
        ax[2].scatter(dates,chls[:,2],marker='o',label='CDOM',color='indigo',s=10)
        ax[2].set_ylabel('Colored Disolved Organic Mather $mg/m^3$')
        ax[2].set_xlabel('year')
        ax[2].legend()
        plt.show()
        
        plot_d_data = [y_data.T[i]  for i in [0,1,2,3,4]]
        plot_l_data = [pred.clone().detach().numpy().T[i] for i in [0,1,2,3,4]]
        corrections =  [0.2,0.25,0.35,0.25,0]
    
        lambdas = [412.5,442.5,490,510,555]
        y_labels = []
        for i in range(5): y_labels.append( '$RRS_{d,\lambda='+str(lambdas[i])+'}(sr^{-1})$')
        start,end = 0,5
        fig, ax = plt.subplots(end - start, 2,width_ratios = [2.5,1],figsize=(15,9))
        for i in range(start,end):

            ks = stats.kstest(plot_d_data[i][~np.isnan(plot_d_data[i])],plot_l_data[i][~np.isnan(plot_d_data[i])])
            corr = stats.pearsonr(plot_d_data[i][~np.isnan(plot_d_data[i])],plot_l_data[i][~np.isnan(plot_d_data[i])])
            rmse = np.sqrt(np.mean((plot_d_data[i][~np.isnan(plot_d_data[i])]-plot_l_data[i][~np.isnan(plot_d_data[i])])**2))
            ax[i-start][0].set_ylabel(y_labels[i])
            ax[i-start][1].set_ylabel('Distribution')
            
            ax[i-start][0].scatter(dates,plot_d_data[i],marker='o',label='Filtered data',color='black',s=10)
            ax[i-start][0].scatter(dates,plot_l_data[i],marker='x',label='Learned data',color='seagreen',s=10,alpha=0.4)
            ax[i-start][1].hist(plot_d_data[i],bins=20,label='Filtered data',density=True,color='black',edgecolor='gray')
            ax[i-start][1].hist(plot_l_data[i][~np.isnan(plot_d_data[i])],bins=20,label='Modeled data',alpha=0.5,density=True,color='seagreen',edgecolor='darkgreen')
            ax[i-start][0].legend(loc='upper right')
            ax[i-start][0].set_xlabel('year')
            ax[i-start][1].legend(loc='upper right')
            ax[i-start][1].set_xlabel(y_labels[i])
            #props = dict(boxstyle='round', facecolor='grey', alpha=0.15)

            my_text1 = "KS Test: {:.3f}, p-value: {:.5E}".format(ks.statistic,ks.pvalue)
            my_text0 = "RMSE: {:.3E}\nCorr: {:.3f}".format(rmse,corr.statistic)
        
            ylimits = ax[i-start][1].get_ylim()
            ax[i-start][1].set_ylim(ylimits[0],ylimits[1]+0.2*(ylimits[1]-ylimits[0]))
            #xlimits = ax[i-start][1].get_xlim()
            #ax[i-start][1].set_xlim(xlimits[0],xlimits[1])#+corrections[i-start]*(xlimits[1]-xlimits[0]))

            ylimits = ax[i-start][0].get_ylim()
            ax[i-start][0].set_ylim(ylimits[0],ylimits[1]+0.2*(ylimits[1]-ylimits[0]))


            ax[i-start][1].text(0.05, 0.95, my_text1, transform=ax[i][1].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)
            ax[i-start][0].text(0.01, 0.95, my_text0, transform=ax[i][0].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)

            xticks =   np.linspace( float(ax[i-start][1].get_xticklabels()[0].get_text()) , float(ax[i-start][1].get_xticklabels()[-1].get_text()),6 )
            xticks_labels = ["{:.3f}".format(label) for label in xticks]
            
            ax[i-start][1].set_xticks(xticks,xticks_labels,fontsize=8,rotation=15)
            fig.tight_layout()
            
        plt.show()

def plot_input(data):
    
    years = np.arange(2006,2013)
    dates = data.dates
    dates = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in dates]
    test_data_init = datetime(year=2000,month=1,day=1) + timedelta(days=test_data.dates[0])
    plot_d_data = [data.x_data[:,i] for i in [4,3,2,1,0]]
    for i in [9,8,7,6,5]: plot_d_data.append(data.x_data[:,i])
    for i in [14,13,12,11,10]: plot_d_data.append(data.x_data[:,i])
    for i in [15,16]: plot_d_data.append(data.x_data[:,i])

    lambdas = [412.5,442.5,490,510,555]
    
    y_labels = []
    for i in [0,1,2,3,4]: y_labels.append( '$R_{RS,\lambda='+str(lambdas[i])+'}(sr^{-1})$')
    for i in [0,1,2,3,4]: y_labels.append( '$E_{dif,\lambda='+str(lambdas[i])+'}(W/m^2)$')
    for i in [0,1,2,3,4]: y_labels.append( '$E_{dir,\lambda='+str(lambdas[i])+'}(W/m^2)$')
    y_labels.append('$Zenith (grad)$')
    y_labels.append('$PAR (W/m^2)$')
    
    
    def plot_partial_plot(start,end):
    
        fig, ax = plt.subplots(end-start, 2,width_ratios = [2.5,1],figsize=(15,9))
        for i in range(start,end):
            ax[i-start][0].set_ylabel(y_labels[i],fontsize=8)
            ax[i-start][1].set_ylabel('Distribution',fontsize=8)

        
            ax[i-start][0].scatter(dates,plot_d_data[i],marker='o',color='gray',s=10)
            #ax[i][0].axvline(t_data_mark,color='gray',ls='--')
            #xlimits  = ax[i][0].get_xlim()
            #ylimits = ax[i][0].get_ylim()
            #ax[i][0].fill_between(np.arange(xlimits[0],xlimits[1]), ylimits[1],y2=ylimits[0],\
                #                  where = np.arange(xlimits[0],xlimits[1]) > t_data_mark, facecolor='gray', alpha=.3,label='Test data')
            #ax[i][0].set_xlim(*xlimits)
            #ax[i][0].set_ylim(*ylimits)
            ax[i-start][0].set_xticklabels(ax[i-start][0].get_xticklabels(), rotation=0, ha='center',fontsize=8)
            ax[i-start][1].hist(plot_d_data[i],bins=20,density=True,color='gray',edgecolor='black',label=y_labels[i])
            ax[i-start][1].set_xticklabels(ax[i-start][1].get_xticklabels(),fontsize=8)
            #ax[i-start][0].legend(loc='upper left')
            ax[i-start][0].set_xlabel('year',fontsize=8)
            ax[i-start][1].legend(loc='upper right')
            ax[i-start][1].set_xlabel(y_labels[i],fontsize=8)
            ax[i-start][0].set_yticklabels(ax[i-start][0].get_yticklabels(),fontsize=8)
            ax[i-start][1].set_yticklabels(ax[i-start][1].get_yticklabels(),fontsize=8)
        plt.show()
    plot_partial_plot(0,5)
    plot_partial_plot(5,10)
    plot_partial_plot(10,15)
    plot_partial_plot(15,17)


def custom_LSELoss2(input,output,nan_array):
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


def plot_losses(shape_layers = [(22,200),(200,9)],epochs=300,learning_rate=1e-6,batch_size=40,loss_fn = custom_LSELoss,NNModel = NNFullyConectedModel,plot_loss = True,images_means=None,images_stds=None,labels_means=None,labels_stds=None,identity=False):
    """ possible NNModels: NNConvolutionalModel, NNFullyConectedModel, AutoencoderNNFullyConectedModel"""
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  #next_iteration_image,next_iteration_label = next(iter(train_dataloader))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    NNModel = NNModel(shape_layers = shape_layers).to("cpu")
    optimizer = torch.optim.Adam(NNModel.parameters(), lr=learning_rate)
    test_loss = []
    train_loss = []
    for epoch in range(epochs):
        train_loss.append(train_loop(train_dataloader, NNModel, loss_fn, optimizer, batch_size,decoder=False,images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds,identity=identity))
        test_loss.append(test_loop(test_dataloader, NNModel, loss_fn,batch_size,decoder=False,images_means=images_means,images_stds=images_stds,labels_means=labels_means,labels_stds=labels_stds,identity=identity))
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
  

init_time = time.time()
data_path = '/Users/carlos/Documents/surface_data_analisis/npy_data'

train_data = customTensorData(data_path,which='Train',without_off_values=False,randomice=True,seed=345,identity=True)
test_data = customTensorData(data_path,which = 'Test',without_off_values=False,randomice=True,seed=345,identity=True)


#train_dataloader,test_dataloader,NNModel=plot_losses(epochs=300,learning_rate=0.01,batch_size=800,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel2,plot_loss = True,\
#                                                    images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds,identity=True)
#plot_output(NNModel,train_data,images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds,identity=True,indexes = test_data.my_indexse)
#torch.save(NNModel.state_dict(),'nngraphs/FullyConnectedIdentityConvolutional.pt')





def plot_fully_conected_saved():
    """
    train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,20),(20,20),(20,20),(20,20),(20,20),(20,9)],epochs=100,learning_rate=0.01,batch_size=800,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = True)
    torch.save(NNModel.state_dict(),'nngraphs/FullyConnected4layersOf20')
    """
    train_data = customTensorData(data_path,which='Train',without_off_values=False,randomice=True,seed=345,identity=False)
    test_data = customTensorData(data_path,which = 'Test',without_off_values=False,randomice=True,seed=345,identity=False)
    NNModel = NNFullyConectedModel(shape_layers = [(17,20),(20,20),(20,20),(20,20),(20,20),(20,9)]).to("cpu")
    NNModel.load_state_dict(torch.load('nngraphs/FullyConnected4layersOf20.pt'))
    NNModel.eval()
    plot_output(NNModel,train_data,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds)

def plot_convolutional_saved():
    """
    train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,20),(20,20),(20,20),(20,20),(20,20),(20,9)],epochs=1100,learning_rate=0.01,batch_size=800,loss_fn = custom_LSELoss2,NNModel =NNConvolutionalModel,plot_loss = True)
    torch.save(NNModel.state_dict(),'nngraphs/ConvolutionalNN')
    """
    
    train_data = customTensorData(data_path,which='Train',without_off_values=False,randomice=True,seed=345,identity=False)
    test_data = customTensorData(data_path,which = 'Test',without_off_values=False,randomice=True,seed=345,identity=False)
    NNModel = NNConvolutionalModel().to("cpu")
    NNModel.load_state_dict(torch.load('nngraphs/ConvolutionalNN.pt'))
    NNModel.eval()
    plot_output(NNModel,train_data,images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds,indexes = test_data.my_indexse)

def plot_identity_saved():
    """
    train_dataloader,test_dataloader,NNModel=plot_losses(epochs=100,learning_rate=0.01,batch_size=800,loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel2,plot_loss = True,\
                                                        images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds,identity=True)
    torch.save(NNModel.state_dict(),'nngraphs/FullyConnectedIdentity.pt')
    this was using the fully connected saved version as input from the first cape of the nn. Has an error of 0.019510559329612914
    """
    train_data = customTensorData(data_path,which='Train',without_off_values=False,randomice=True,seed=345,identity=True)
    test_data = customTensorData(data_path,which = 'Test',without_off_values=False,randomice=True,seed=345,identity=True)
    
    NNModel = NNFullyConectedModel2()
    NNModel.load_state_dict(torch.load('nngraphs/FullyConnectedIdentity.pt'))
    NNModel.eval()
    plot_output(NNModel,train_data,images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds,identity=True,indexes = test_data.my_indexse)

plot_identity_saved()




#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = None,epochs=500,learning_rate=0.01,batch_size=800,loss_fn = custom_LSELoss2,NNModel = NNConvolutionalModel,plot_loss = True)
#plot_output(NNModel,train_data,images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds)

#train_dataloader,test_dataloader,NNModel=plot_losses(shape_layers = [(17,20+21),(20+21,3)],epochs=100,learning_rate=0.01,batch_size=800,\
#                                                         loss_fn = custom_LSELoss2,NNModel = NNFullyConectedModel,plot_loss = True,\
#                                                         images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds)

#print(NNModel.perturbation_factors)

#plot_output(NNModel,train_data,images_means=train_data.x_normalization_means,images_stds=train_data.x_normalization_stds,labels_means=train_data.y_normalization_means,labels_stds=train_data.y_normalization_stds)
