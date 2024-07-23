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
from torch.utils.data import DataLoader,random_split

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import tempfile
from pathlib import Path
from functools import partial
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS

from datetime import datetime, timedelta
import pandas as pd
import os
import scipy
from scipy import stats
import time
import torch.distributed as dist
import sys

import warnings
import Forward_module as fm
import read_data_module as rdm


class NN_first_layer(nn.Module):

    def __init__(self,precision = torch.float32,input_layer_size=10,output_layer_size=10,number_hiden_layers = 1,dim_hiden_layers = 20,non_linearity = nn.CELU(),alpha=1.):
        super().__init__()
        self.flatten = nn.Flatten()

        linear_celu_stack = []
        input_size = input_layer_size
        if hasattr(dim_hiden_layers, '__iter__'):
            output_size = dim_hiden_layers[0]
        else:
            output_size = dim_hiden_layers
            
        for hl in range(number_hiden_layers):
            linear_celu_stack += [nn.Linear(input_size,output_size),nn.CELU(alpha=alpha)]
            if hl < (number_hiden_layers-1):
                input_size = output_size
                if hasattr(dim_hiden_layers, '__iter__'):
                    output_size = dim_hiden_layers[hl+1]
                else:
                    output_size = dim_hiden_layers
        linear_celu_stack += [nn.Linear(output_size,output_layer_size),nn.CELU(alpha=alpha)]
            

        self.linear_celu_stack = nn.Sequential( *linear_celu_stack  )

    def forward(self, x):
        x = self.linear_celu_stack(x)
        return x


def train_one_epoch(epoch_index,training_dataloader,loss_fn,optimizer,model,batch_size = 20):
       
    running_loss = 0.

    for i, data in enumerate(training_dataloader):
        # Every data instance is an input + label pair
        inputs, labels_nan = data
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        outputs = torch.masked_fill(outputs,torch.isnan(labels_nan),0)
        labels = torch.masked_fill(labels_nan,torch.isnan(labels_nan),0)

        # Compute the loss and its gradients

        loss = loss_fn(torch.flatten(outputs,1), torch.flatten(labels,1),torch.flatten(labels_nan,1))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss
    running_loss /= (i+1)
    return running_loss.item()

def validation_loop(epoch_index,validation_dataloader,loss_fn,optimizer,model,batch_size = 20):

    
    running_vloss = 0.
    with torch.no_grad():
        for i, vdata in enumerate(validation_dataloader):
            vinputs, vlabels_nan = vdata
            voutputs = model(vinputs)
            voutputs = torch.masked_fill(voutputs,torch.isnan(vlabels_nan),0)
            vlabels = torch.masked_fill(vlabels_nan,torch.isnan(vlabels_nan),0)
            vloss = loss_fn(torch.flatten(voutputs,1), torch.flatten(vlabels,1),torch.flatten(vlabels_nan,1))
            running_vloss += vloss

        running_vloss /= (i + 1)
        
    return running_vloss.item()


class composed_loss_function(nn.Module):

    def __init__(self,precision = torch.float32):
        super(composed_loss_function, self).__init__()
        
    def forward(self,pred_,Y_obs,nan_array):
        lens = torch.tensor([len(element[~element.isnan()])  for element in nan_array])
        obs_error = torch.trace(   ((pred_ - Y_obs)  @ (pred_ - Y_obs ).T ))
        return obs_error



def train_cifar(config,data_dir = None):
    time_init = time.time()

    my_device = 'cpu'


    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    
    train_data = rdm.customTensorData(data_path=data_dir,which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=my_device)
 
    batch_size = int(config['batch_size'])
    number_hiden_layers = config['number_hiden_layers']
    dim_hiden_layers = config['dim_hiden_layers'] 
    lr = config['lr']
    betas1 = config['betas1'] 
    betas2 = config['betas2']
    mean_validation_loss = 0.
    mean_train_loss = 0.

    for i in range(10):

        train_d,validation = random_split(train_data,[0.95,0.05],generator = torch.Generator().manual_seed(i))
        training_dataloader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation,batch_size = batch_size,shuffle=True)

        model = NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
                       number_hiden_layers = number_hiden_layers,dim_hiden_layers = dim_hiden_layers,non_linearity = nn.CELU()).to(my_device)


        loss_function = composed_loss_function()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas1,betas2))

        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                model.load_state_dict(checkpoint_state["state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0

        validation_loss = []
        train_loss = []
        for epoch in range(start_epoch,10):
            train_loss.append(train_one_epoch(epoch,training_dataloader,loss_function,optimizer,model,batch_size = int(batch_size)))
            validation_loss.append(validation_loop(epoch,validation_dataloader,loss_function,optimizer,model,batch_size = int(batch_size)))
            
        mean_validation_loss += validation_loss[-1]
        mean_train_loss += train_loss[-1]
        
    checkpoint_data = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)

        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            {"loss_validation": mean_validation_loss/10,'loss_train':mean_train_loss/10},
            checkpoint=checkpoint,
        )
    print('Finishe Training {} seconds'.format(time.time() - time_init))

def test_accuracy(model, device="cpu",data_dir = None,config = None):

    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir + '/../cst.csv',my_device = device)
    
    test_data = rdm.customTensorData(data_path=data_dir,which='test',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=device)
    test_dataloader = DataLoader(test_data,batch_size = batch_size,shuffle=True)

    batch_size = config['batch_size']
    number_hiden_layers = config['number_hiden_layers']
    dim_hiden_layers = config['dim_hiden_layers'] 
    lr = config['lr']
    betas1 = config['betas1'] 
    betas2 = config['betas2']

    model = NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
                       number_hiden_layers = number_hiden_layers,dim_hiden_layers = dim_hiden_layers,non_linearity = nn.CELU()).to(device)


    loss_function = composed_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas1,betas2))

    total = 0
    with torch.no_grad():
        loss = validation_loop(0,test_dataloader,loss_function,optimizer,model,batch_size = int(batch_size))

    return loss


if __name__ == '__main__':
    
    data_dir = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
   
    torch.manual_seed(0)
    
    """
    batch_size = config['batch_size'] #50
    number_hiden_layers = config['number_hiden_layers'] #1
    dim_hiden_layers = config['dim_hiden_layers'] #20
    lr = config['lr'] #0.001
    betas1 = config['betas1'] #0.9
    betas2 = config['betas2'] # 0.999
    """
    
    config = {
        "batch_size": tune.choice(np.arange(20,210,10)),
        "number_hiden_layers": tune.choice(np.arange(1,10)),
        "dim_hiden_layers":tune.choice(np.arange(20,30)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "betas1": tune.choice(np.linspace(0.8,0.999,10)),
        "betas2": tune.choice(np.linspace(0.9,0.9999,10))
    }

    config_space = CS.ConfigurationSpace({
            "batch_size":list(np.arange(20,210,10)),
            "number_hiden_layers":list(np.arange(1,10)),
            "dim_hiden_layers":list(np.arange(20,30)),
            "betas1":list(np.linspace(0.8,0.999,10)),
            "betas2":list(np.linspace(0.8,0.9999,10)),
            "lr":list(np.logspace(-4,-1,10))
        })



    max_iterations = 10
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iterations,
        reduction_factor=2,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB(
         space=config_space,  metric="loss_validation", mode='min'
    )
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)


    tuner = tune.Tuner(
        partial(train_cifar, data_dir=data_dir),
        run_config=train.RunConfig(
            name="bohb_minimization", stop={"training_iteration": max_iterations}
        ),
        tune_config=tune.TuneConfig(
            metric="loss_validation",
            mode="min",
            scheduler=bohb_hyperband,
            search_alg=bohb_search,
            num_samples=300,
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result("loss_validation","min")  # Get best result object

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss_validation"]))


    my_device = 'cpu'

    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    train_data = rdm.customTensorData(data_path=data_dir,which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=my_device)
 
    batch_size = best_result.config['batch_size']
    number_hiden_layers = best_result.config['number_hiden_layers']
    dim_hiden_layers = best_result.config['dim_hiden_layers'] 
    lr = best_result.config['lr']
    betas1 = best_result.config['betas1'] 
    betas2 = best_result.config['betas2']

    model = NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
                       number_hiden_layers = number_hiden_layers,dim_hiden_layers = dim_hiden_layers,non_linearity = nn.CELU()).to(my_device)


    loss_function = composed_loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas1,betas2))
    
    
    validation_loss = []
    train_loss = []
    for epoch in range(300):
        train_loss.append(train_one_epoch(epoch,train_data,loss_function,optimizer,model,batch_size = int(batch_size)))
        validation_loss.append(validation_loop(epoch,train_data,loss_function,optimizer,model,batch_size = int(batch_size)))
    
    #plt.plot(validation_loss,label='validation_loss')
    #plt.plot(train_loss,label='train_loss')
    #plt.legend()
    #plt.show()

    torch.save(model.state_dict(), data_dir+'/../VAE_model/model_first_part.pt')
    
    
