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

    def __init__(self,precision = torch.float32,input_layer_size=10,output_layer_size=10,number_hiden_layers = 1,\
                 dim_hiden_layers = 20,alpha=1.,dim_last_hiden_layer = 1):
        super().__init__()
        self.flatten = nn.Flatten()

        linear_celu_stack = []
        input_size = input_layer_size
        if hasattr(dim_hiden_layers, '__iter__'):
            output_size = dim_hiden_layers[0]
        else:
            output_size = dim_hiden_layers
            
        for hl in range(number_hiden_layers):
            if hl != (number_hiden_layers - 1):
                linear_celu_stack += [nn.Linear(input_size,output_size),nn.CELU(alpha=alpha)]
                input_size = output_size
                if hasattr(dim_hiden_layers, '__iter__'):
                    output_size = dim_hiden_layers[hl+1]
                else:
                    output_size = dim_hiden_layers
            else:
                linear_celu_stack += [nn.Linear(input_size,dim_last_hiden_layer),nn.CELU(alpha=alpha)]
                    
            
        linear_celu_stack += [nn.Linear(dim_last_hiden_layer,output_layer_size),nn.CELU(alpha=alpha)]
            

        self.linear_celu_stack = nn.Sequential( *linear_celu_stack  )

    def forward(self, x):
        x = self.linear_celu_stack(x)
        return x


def train_one_epoch(epoch_index,training_dataloader,loss_fn,optimizer,model,dates=None):
       
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

        loss = loss_fn(outputs[:,0,:], labels[:,0,:],labels_nan[:,0,:],model.parameters())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss
    running_loss /= (i+1)
    return running_loss.item()

def validation_loop(epoch_index,validation_dataloader,loss_fn,optimizer,model):

    
    running_vloss = 0.
    with torch.no_grad():
        for i, vdata in enumerate(validation_dataloader):
            vinputs, vlabels_nan = vdata
            voutputs = model(vinputs)
            voutputs = torch.masked_fill(voutputs,torch.isnan(vlabels_nan),0)
            vlabels = torch.masked_fill(vlabels_nan,torch.isnan(vlabels_nan),0)
            vloss = loss_fn(torch.flatten(voutputs,1), torch.flatten(vlabels,1),torch.flatten(vlabels_nan,1),model.parameters())
            running_vloss += vloss

        running_vloss /= (i + 1)
        
    return running_vloss.item()


class composed_loss_function(nn.Module):

    def __init__(self,precision = torch.float32,l1_regularization = 0.001):
        super(composed_loss_function, self).__init__()
        self.l1_regularization = l1_regularization
        
    def forward(self,pred_,Y_obs,nan_array,parameters,dates = None):
        lens = torch.tensor([len(element[~element.isnan()])  for element in nan_array])
        obs_error = torch.trace(   ((pred_ - Y_obs) @ (pred_ - Y_obs ).T)/lens )

        l1_norm = sum(torch.linalg.norm(p, 1) for p in parameters)
        
        return obs_error + self.l1_regularization*l1_norm



def train_cifar(config,data_dir = None):
    time_init = time.time()

    my_device = 'cpu'


    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    
    train_data = rdm.customTensorData(data_path=data_dir,which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')
 
    batch_size = int(config['batch_size'])
    number_hiden_layers = config['number_hiden_layers']
    dim_hiden_layers = config['dim_hiden_layers'] 
    lr = config['lr']
    betas1 = config['betas1'] 
    betas2 = config['betas2']
    dim_last_hiden_layer = config['last_layer_size']
    alpha = config['alpha']
    l1_regularization = config['l1_regularization']
    
    mean_validation_loss = 0.
    mean_train_loss = 0.

    for i in range(5):

        train_d,validation = random_split(train_data,[0.95,0.05],generator = torch.Generator().manual_seed(i))
        training_dataloader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation,batch_size = batch_size,shuffle=True)

        model = NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
                               number_hiden_layers = number_hiden_layers,dim_hiden_layers = dim_hiden_layers,dim_last_hiden_layer = dim_last_hiden_layer,alpha = alpha).to(my_device)


        loss_function = composed_loss_function(l1_regularization = l1_regularization)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas1,betas2),maximize=False)

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
        for epoch in range(start_epoch,30):
            train_loss.append(train_one_epoch(epoch,training_dataloader,loss_function,optimizer,model))
            validation_loss.append(validation_loop(epoch,validation_dataloader,loss_function,optimizer,model))
            
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
            {"loss_validation": mean_validation_loss/5,'loss_train':mean_train_loss/5},
            checkpoint=checkpoint,
        )
    print('Finishe Training {} seconds'.format(time.time() - time_init))

def test_accuracy(model, device="cpu",data_dir = None,config = None):

    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir + '/../cst.csv',my_device = device)
    


    batch_size = int(config['batch_size'])
    number_hiden_layers = config['number_hiden_layers']
    dim_hiden_layers = config['dim_hiden_layers']
    dim_last_hiden_layer = config['last_layer_size']
    lr = config['lr']
    betas1 = config['betas1'] 
    betas2 = config['betas2']
    alpha = config['alpha']
    l1_regularization = config['l1_regularization']

    test_data = rdm.customTensorData(data_path=data_dir,which='test',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=device,normilized_NN='scaling')
    test_dataloader = DataLoader(test_data,batch_size = batch_size,shuffle=True)
    

    model = NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
                           number_hiden_layers = number_hiden_layers,dim_hiden_layers = dim_hiden_layers,dim_last_hiden_layer = dim_last_hiden_layer,alpha=alpha).to(device)


    loss_function = composed_loss_function(l1_regularization = l1_regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas1,betas2),maximize=False)

    total = 0
    with torch.no_grad():
        loss = validation_loop(0,test_dataloader,loss_function,optimizer,model)

    return loss


def explore_hyperparameters():
    
    data_dir = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
   
    torch.manual_seed(0)

    config_space = CS.ConfigurationSpace({
        "batch_size":list(np.arange(10,41,4)),
        "number_hiden_layers":list(np.arange(1,10)),
        "dim_hiden_layers":list(np.arange(10,30)),
        "betas1":CS.Float("betas1",bounds=(0.5, 0.99),distribution=CS.Normal(0.7, 0.5)),
        "betas2":CS.Float("betas2",bounds=(0.5, 0.99),distribution=CS.Normal(0.7, 0.5)),
        "lr":CS.Float("lr",bounds=(0.0001, 0.01),distribution=CS.Normal(0.002, 0.01),log=True),
        "last_layer_size":list(np.arange(10,30)) ,
        "alpha":CS.Float("alpha",bounds=(0.5, 3),distribution=CS.Normal(1, 0.25)),
        "l1_regularization":CS.Float("l1_regularization",distribution = CS.Normal(0.001,0.01),bounds = (0,0.1))
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
            num_samples=200,
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result("loss_validation","min")  # Get best result object

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss_validation"]))

    
    

def save_cvae_first_part():
    experiment_path = '/Users/carlos/ray_results/bohb_minimization'
    data_dir = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
        
    restored_tuner = tune.Tuner.restore(experiment_path, trainable=partial(train_cifar, data_dir=data_dir))
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result("loss_validation","min")
 
    batch_size = int(best_result.config['batch_size'])
    number_hiden_layers = best_result.config['number_hiden_layers']
    dim_hiden_layers = best_result.config['dim_hiden_layers']
    dim_last_hiden_layer = best_result.config['last_layer_size']
    lr = best_result.config['lr']
    betas1 = best_result.config['betas1'] 
    betas2 = best_result.config['betas2']
    alpha = best_result.config['alpha']
    l1_regularization = best_result.config['l1_regularization']

    my_device = 'cpu'

    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    train_data = rdm.customTensorData(data_path=data_dir,which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
                           number_hiden_layers = number_hiden_layers,dim_hiden_layers = dim_hiden_layers,dim_last_hiden_layer = dim_last_hiden_layer,alpha=alpha).to(my_device)

    
    loss_function = composed_loss_function(l1_regularization = l1_regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas1,betas2),maximize=False)
    
    
    validation_loss = []
    train_loss = []
    for epoch in range(50):
        train_loss.append(train_one_epoch(epoch,train_dataloader,loss_function,optimizer,model,dates = train_data.dates))

    torch.save(model.state_dict(), data_dir+'/../VAE_model/model_first_part.pt')

    
if __name__ == "__main__":

    explore_hyperparameters()
    save_cvae_first_part()
    
    experiment_path = '/Users/carlos/ray_results/bohb_minimization'
    data_dir = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
        
    restored_tuner = tune.Tuner.restore(experiment_path, trainable=partial(train_cifar, data_dir=data_dir))
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result("loss_validation","min")
 
    batch_size = int(best_result.config['batch_size'])
    number_hiden_layers = best_result.config['number_hiden_layers']
    dim_hiden_layers = best_result.config['dim_hiden_layers']
    dim_last_hiden_layer = best_result.config['last_layer_size']
    lr = best_result.config['lr']
    betas1 = best_result.config['betas1'] 
    betas2 = best_result.config['betas2']
    alpha = best_result.config['alpha']
    l1_regularization = best_result.config['l1_regularization']

    my_device = 'cpu'

    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    data = rdm.customTensorData(data_path=data_dir,which='all',per_day = False,randomice=False,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')
    dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)

    model = NN_first_layer(precision = torch.float32,input_layer_size=17,output_layer_size=9,\
                           number_hiden_layers = number_hiden_layers,dim_hiden_layers = dim_hiden_layers,dim_last_hiden_layer = dim_last_hiden_layer,alpha=alpha).to(my_device)
    
    model.load_state_dict(torch.load(data_dir+'/../VAE_model/model_first_part.pt'))
    model.eval()

    X,Y = next(iter(dataloader))
    pred = model(X)

    chla = np.exp(Y[:,0,0]*(data.y_max[0] - data.y_min[0]) + data.y_min[0])
    chla_pred = np.exp(pred[:,0,0].clone().detach()*(data.y_max[0] - data.y_min[0]) + data.y_min[0])
    plt.plot(data.dates,chla,'o',color='black',label ='chlorophyll measured')
    plt.plot(data.dates,chla_pred,'o',color='blue',label = 'chlorophyll predicted by nn')
    plt.legend()
    plt.show()
    
