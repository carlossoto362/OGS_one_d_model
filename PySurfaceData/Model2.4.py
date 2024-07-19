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
from torch.distributions.multivariate_normal import MultivariateNormal
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
        self.batch_size = batch_size
        
        self.conv1 = nn.Conv1d(1, 5, 5, stride = 1, padding = 1)
        self.conv2 = nn.Conv1d(5, 5, 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv1d(5, 5, 5, stride = 1, padding = 1)
        self.CELU = nn.CELU()
        self.ReLU = nn.ReLU()
        self.Linear1 = nn.Linear(55,20)
        self.Linear2 = nn.Linear(20,20)
        self.Linear3 = nn.Linear(20,20)
        self.Linear4 = nn.Linear(20,20)
        self.Linear5 = nn.Linear(20,12)

        self.Forward_Model = Forward_Model(learning_chla = False, learning_perturbation_factors = True)



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
        y = torch.flatten(y,1)
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
        
        sigma = y[:,3:].reshape((y.shape[0],3,3)).unsqueeze(1)
        y_hat_mean = y[:,:3].unsqueeze(1)
        covariance_matrix = torch.transpose(sigma,2,3) @ sigma

        normal_samples = MultivariateNormal(torch.zeros(3),torch.eye(3)).sample(sample_shape=torch.Size([y.shape[0],1])).\
            repeat_interleave(3,2,output_size = 9).reshape(y.shape[0],1,3,3)#trick for matmul with different covariances. 

        y_sample = y_hat_mean + (sigma @ normal_samples )[:,:,:,0]
        
        y_use = y_sample.clone()
        y_use[:,:,0] = y_sample[:,:,0] * self.y_std[0] + self.y_mean[0]


        X = self.rearange_RRS(image)
        rrs_ = self.Forward_Model(X,parameters = y_use,constant = self.constant)
        rrs_ = (rrs_ - self.x_mean[:5])/self.x_std[:5]
 
        kd_ = kd(9.,X[:,:,0],X[:,:,1],X[:,:,2],X[:,:,3],X[:,:,4],torch.exp(y_use[:,:,0]),torch.exp(y_use[:,:,1]),torch.exp(y_use[:,:,2]),self.Forward_Model.perturbation_factors,constant = self.constant)
        kd_ = (kd_  - self.y_mean[1:6])/self.y_std[1:6]

        bbp_ = bbp(X[:,:,0],X[:,:,1],X[:,:,2],X[:,:,3],X[:,:,4],torch.exp(y_use[:,:,0]),torch.exp(y_use[:,:,1]),torch.exp(y_use[:,:,2]),self.Forward_Model.perturbation_factors,constant = self.constant)[:,[1,2,4]]
        bbp_ = (bbp_ - self.y_mean[6:])/self.y_std[6:9]

        y_hat_mean[:,:,0] = y_hat_mean[:,:,0] * self.y_std[0] + self.y_mean[0]
        y_hat_mean = y_hat_mean[:,0,:]
        covariance_matrix = covariance_matrix[:,0,:,:] * torch.tensor([[self.y_std[0]**2,self.y_std[0],self.y_std[0]],[self.y_std[0],1,1],[self.y_std[0],1,1]])

        return y_sample[:,0,:],covariance_matrix,y_hat_mean,kd_,bbp_,rrs_

class composed_loss_function(nn.Module):

    def __init__(self,precision = torch.float32,my_device = 'cpu',rrs_std=torch.ones(5),chla_std=torch.ones(1),kd_std=torch.ones(5),bbp_std=torch.ones(3)):
        super(composed_loss_function, self).__init__()
        self.precision = precision
        self.x_a = torch.zeros(3)
        self.s_a = torch.eye(3)*4.9
        self.s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)
        rrs_cov = self.s_e * (rrs_std.unsqueeze(0).T@rrs_std.unsqueeze(0))
        self.rrs_cov_inv = rrs_cov.inverse().to(torch.float32)

        Y_cov = torch.empty(9)
        Y_cov[0] = chla_std
        Y_cov[1:6] = kd_std
        Y_cov[6:] = bbp_std


        Y_cov = torch.eye(9) * (Y_cov.unsqueeze(0).T@Y_cov.unsqueeze(0))
        self.Y_cov_inv = Y_cov.inverse().to(torch.float32)


    def forward(self,pred_,Y_obs,rrs_,rrs_pred,nan_array, param):

        rrs_error = torch.trace(   (rrs_ - rrs_pred) @ ( self.rrs_cov_inv @ (rrs_ - rrs_pred ).T ) )
        
        #lens = torch.tensor([len(element[~element.isnan()])  for element in nan_array])
        obs_error = torch.trace(   ((pred_ - Y_obs) @ ( self.Y_cov_inv @ (pred_ - Y_obs ).T )))

        return rrs_error + obs_error + 0.001*param.mean()

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
        model.batch_size = X.shape[0]
        Y_composed = torch.masked_fill(Y,torch.isnan(Y),0)
        
        loss = 0
        for _ in range(1):
            chla_,covariance_matrix,chla_hat_mean,kd_,bbp_,rrs_= model(X)
            pred_ = torch.zeros(Y_composed.shape[0],Y_composed.shape[2])

            pred_[:,1:6] = torch.masked_fill(kd_,torch.isnan(Y[:,0,1:6]),0)
            pred_[:,6:] = torch.masked_fill(bbp_,torch.isnan(Y[:,0,6:9]),0)
            pred_[:,0] = torch.masked_fill(chla_[:,0],torch.isnan(Y[:,0,0]),0)
            loss += loss_fn(pred_,Y_composed[:,0,:],rrs_,X[:,0,:5],Y,chla_hat_mean)
        loss /= 1
        
        perturbation_factors_history = list(iter(model.parameters()))[-1].clone().detach().numpy()
        chla_history[k:k+len(X)] = chla_hat_mean.clone().detach().numpy()
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
            model.batch_size = X.shape[0]
            Y_composed = torch.masked_fill(Y,torch.isnan(Y),0)
            pred_ = torch.zeros(Y_composed.shape[0],Y_composed.shape[2])
            
            chla_,covariance_matrix,chla_hat_mean,kd_,bbp_,rrs_= model(X)

            pred_[:,1:6] = torch.masked_fill(kd_,torch.isnan(Y[:,0,1:6]),0)
            pred_[:,6:] = torch.masked_fill(bbp_,torch.isnan(Y[:,0,6:9]),0)
            pred_[:,0] = torch.masked_fill(chla_[:,0],torch.isnan(Y[:,0,0]),0)
            loss_ = loss_fn(pred_,Y_composed[:,0,:],rrs_,X[:,0,:5],Y,chla_hat_mean)
            
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
        if j %10 == 0 :
            print(j,'done')
        
    if save == True:
        np.save(save_path + '/train_loss_VAE2.npy',train_loss)
        np.save(save_path + '/test_loss_VAE2.npy',test_loss)
        np.save(save_path + '/chla_history_VAE2.npy',chla_history)
        np.save(save_path + '/kd_history_VAE2.npy',kd_history)
        np.save(save_path + '/bbp_history_VAE2.npy',bbp_history)
        np.save(save_path + '/perturbation_factors_history_VAE2.npy',perturbation_factors_history)

    return train_loss,test_loss,chla_history,kd_history,bbp_history,perturbation_factors_history

data_path = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
my_device = torch.device("mps")
my_device = 'cpu'
constant = read_constants(file1='cte_lambda.csv',file2='cst.csv',my_device = my_device)
data = customTensorData(data_path=data_path,which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853)
test_data =  customTensorData(data_path=data_path,which='test',per_day = False,randomice=True,one_dimensional = True,seed = 1853)
batch_size = data.len_data
chla_mean = data.y_mean[0]
chla_std = data.y_std[0]


lr_ = 0.01

loss_fn = composed_loss_function(rrs_std=torch.tensor(data.x_std[:5]),chla_std=torch.tensor(data.y_std[0]),kd_std=torch.tensor(data.y_std[1:6]),bbp_std=torch.tensor(data.x_std[6:9]))
model = NNConvolutionalModel(constant = constant,x_mean = data.x_mean,x_std = data.x_std, y_mean = data.y_mean, y_std = data.y_std,batch_size = batch_size)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=test_data.len_data, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(),lr=lr_)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
epochs = 501
    
train_loss,test_loss,chla_history,kd_history,bbp_history,perturbation_factors_history = get_training_history(dataloader,test_dataloader,model,loss_fn,optimizer,\
                                                                                                            epochs,batch_size,save=True,save_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data/')
print(lr_,test_loss[-1],train_loss[-1])

torch.save(model.state_dict(),'/Users/carlos/Documents/OGS_one_d_model/plot_data/model_state_dict_VAE2.npy')


data = customTensorData(data_path=data_path,which='all',per_day = False,randomice=False,one_dimensional = True,seed = 1853)
dataloader = DataLoader(data, batch_size=data.len_data, shuffle=False)

 
load_path = '/Users/carlos/Documents/OGS_one_d_model/plot_data'
model.load_state_dict(torch.load(load_path + '/model_state_dict_VAE2.npy'))
model.eval()


#train_loss = np.load(load_path + '/train_loss_VAE.npy')
#test_loss = np.load(load_path + '/test_loss_VAE.npy')
#perturbation_factors_history = np.load(load_path + '/perturbation_factors_history_VAE.npy')

#plot_tracked_parameters(perturbation_factors_history = perturbation_factors_history, test_loss = test_loss, train_loss = train_loss, save_file_name = '/perturbation_factors_VAE.pdf',save=True,color_palet=1,side = 'left',with_loss =False)

X,Y = next(iter(dataloader))
chla_,covariance_matrix,chla_hat_mean,kd_,bbp_,rrs_= model(X)
X = model.rearange_RRS(X)

rrs_hat = rrs_ * model.x_std[:5] + model.x_mean[:5]


def save_var_uncertainties(model, X, chla_hat_mean, covariance_matrix,rrs_hat, constant = None, save_path ='/Users/carlos/Documents/surface_data_analisis/LastVersion/results_VAE2_VAEparam',dates = [] ):
    parameters_eval = chla_hat_mean.unsqueeze(1)
    evaluate_model = evaluate_model_class(model.Forward_Model,X,constant=constant)
        
    X_hat = np.empty((len(parameters_eval),6))
    X_hat[:,::2] = chla_hat_mean.clone().detach()
    X_hat[:,1::2] = torch.sqrt(torch.diagonal(covariance_matrix,dim1=1,dim2=2).clone().detach())
       
    kd_hat = torch.empty((len(parameters_eval),10))
    bbp_hat = torch.empty((len(parameters_eval),6))
    bbp_index = 0

    kd_values = evaluate_model.kd_der(parameters_eval)
    kd_derivative = torch.empty((len(parameters_eval),5,3))
    kd_derivative_ = torch.autograd.functional.jacobian(evaluate_model.kd_der,inputs=(parameters_eval))
    
    for i in range(len(parameters_eval)):
        kd_derivative[i] = torch.reshape(kd_derivative_[i,:,i,:,:],(5,3))
               
    kd_delta = error_propagation(kd_derivative,covariance_matrix)
    kd_hat[:,::2] = kd_values.clone().detach()
    kd_hat[:,1::2] = torch.sqrt(kd_delta).clone().detach()
    bbp_values = evaluate_model.bbp_der(parameters_eval)

    bbp_derivative = torch.empty((len(parameters_eval),3,3))
    bbp_derivative_ = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))
    for i in range(len(parameters_eval)):
        bbp_derivative[i] = torch.reshape(bbp_derivative_[i,:,i,:,:],(3,3))
        
    bbp_delta = error_propagation(bbp_derivative,covariance_matrix)
    bbp_hat[:,::2] = bbp_values.clone().detach()
    bbp_hat[:,1::2] = torch.sqrt(bbp_delta).clone().detach()

    rrs_hat = rrs_hat.clone().detach() 

    np.save(save_path + '/X_hat.npy',X_hat)
    np.save(save_path + '/RRS_hat.npy',rrs_hat)
    np.save(save_path + '/kd_hat.npy',kd_hat)
    np.save(save_path + '/bbp_hat.npy',bbp_hat)
    np.save(save_path + '/dates.npy',dates)
    
save_var_uncertainties(model,X,chla_hat_mean,covariance_matrix,rrs_hat,constant=constant,dates = data.dates)

#plt.plot(data.dates,chla_[:,0],label='prediction')
#plt.plot(data.dates,data.y_normilized[:,0],label='data')
#plt.legend()
#plt.show()

#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_VAE_VAEparam/X_hat.npy',chla_)
#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_VAE_VAEparam/RRS_hat.npy',rrs_)
#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_VAE_VAEparam/kd_hat.npy',kd_)
#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_VAE_VAEparam/bbp_hat.npy',bbp_)


#np.save('/Users/carlos/Documents/surface_data_analisis/LastVersion/results_VAE_VAEparam/dates.npy',data.dates)







    





    
