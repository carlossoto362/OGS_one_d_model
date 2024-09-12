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
from CVAE_model_part_one import NN_first_layer
from torch.linalg import inv 

if 'OGS_ONE_D_HOME_PATH' in os.environ:
    MODEL_HOME = os.environ["OGS_ONE_D_HOME_PATH"]
else:
        
    print("Missing local variable OGS_ONE_D_HOME_PATH. \nPlease add it with '$:export OGS_ONE_D_HOME_PATH=path/to/ogs/one/d/model'.")
    sys.exit()

class NN_second_layer(nn.Module):

    def __init__(self,precision = torch.float32,output_layer_size_mean=3,number_hiden_layers_mean = 1,\
                 dim_hiden_layers_mean = 20,alpha_mean=1,dim_last_hiden_layer_mean = 1,output_layer_size_cov=9,number_hiden_layers_cov = 1,\
                 dim_hiden_layers_cov = 20,alpha_cov=1,dim_last_hiden_layer_cov = 1,x_mul=None,x_add=None,\
                 y_mul=None,y_add=None,constant = None,model_dir='/Users/carlos/Documents/OGS_one_d_model/VAE_model',my_device='cpu',chla_centered=True ):
        super().__init__()

        self.chla_centered = chla_centered
        self.LogSigmoid = nn.LogSigmoid() # for claping
        self.flatten = nn.Flatten()

        self.my_device = my_device
        self.precision = precision
        
        def read_NN_first_layer(model_dir= model_dir):
            """
            read rirst layer trained with in situ data. This layer is made to facilitate the learning of
            the latent such that they are close to in-situ observations.
            """
            model_first_layer = NN_first_layer(precision = self.precision,input_layer_size=17,output_layer_size=9,\
                           number_hiden_layers = 1,dim_hiden_layers = 20,dim_last_hiden_layer = 22,alpha=0.8978238833058).to(self.my_device)
            model_first_layer.load_state_dict(torch.load(model_dir+'/model_first_part.pt'))
            model_first_layer.eval()
            for param in model_first_layer.parameters():
                param.requires_grad = False
            return model_first_layer
        self.first_layer = read_NN_first_layer(model_dir)
        

        linear_celu_stack_mean = []
        input_size = 9
        if hasattr(dim_hiden_layers_mean, '__iter__'):
            output_size = dim_hiden_layers_mean[0]
        else:
            output_size = dim_hiden_layers_mean
            
        for hl in range(number_hiden_layers_mean):
            if hl != (number_hiden_layers_mean - 1):
                linear_celu_stack_mean += [nn.Linear(input_size,output_size),nn.CELU(alpha=alpha_mean)]
                input_size = output_size
                if hasattr(dim_hiden_layers_mean, '__iter__'):
                    output_size = dim_hiden_layers_mean[hl+1]
                else:
                    output_size = dim_hiden_layers_mean
            else:
                linear_celu_stack_mean += [nn.Linear(input_size,dim_last_hiden_layer_mean),nn.CELU(alpha=alpha_mean)]  
        linear_celu_stack_mean += [nn.Linear(dim_last_hiden_layer_mean,output_layer_size_mean),nn.CELU(alpha=alpha_mean)]
        self.linear_celu_stack_mean = nn.Sequential( *linear_celu_stack_mean  )

        linear_celu_stack_cov = []
        input_size = 9
        if hasattr(dim_hiden_layers_cov, '__iter__'):
            output_size = dim_hiden_layers_cov[0]
        else:
            output_size = dim_hiden_layers_cov
            
        for hl in range(number_hiden_layers_cov):
            if hl != (number_hiden_layers_cov - 1):
                linear_celu_stack_cov += [nn.Linear(input_size,output_size),nn.CELU(alpha=alpha_cov)]
                input_size = output_size
                if hasattr(dim_hiden_layers_cov, '__iter__'):
                    output_size = dim_hiden_layers_cov[hl+1]
                else:
                    output_size = dim_hiden_layers_cov
            else:
                linear_celu_stack_cov += [nn.Linear(input_size,dim_last_hiden_layer_cov),nn.CELU(alpha=alpha_cov)]  
        linear_celu_stack_cov += [nn.Linear(dim_last_hiden_layer_cov,output_layer_size_cov),nn.CELU(alpha=alpha_cov)]
        self.linear_celu_stack_cov = nn.Sequential( *linear_celu_stack_cov  )

        self.x_mul = torch.tensor(x_mul).to(self.precision).to(self.my_device)
        self.y_mul = torch.tensor(y_mul).to(self.precision).to(self.my_device)
        self.x_add = torch.tensor(x_add).to(self.precision).to(self.my_device)
        self.y_add = torch.tensor(y_add).to(self.precision).to(self.my_device)

        self.Forward_Model = fm.Forward_Model(learning_chla = False, learning_perturbation_factors = True)
        self.bbp = fm.bbp
        self.kd = fm.kd
        self.constant = constant

    def rearange_RRS(self,x):
        lambdas = torch.tensor([412.5,442.5,490.,510.,555.])
        x_ = x*self.x_mul + self.x_add
        output = torch.empty((len(x),5,5))
        output[:,:,0] = x_[:,0,5:10]
        output[:,:,1] = x_[:,0,10:15]
        output[:,:,2] = lambdas
        output[:,:,3] = x_[:,:,15]
        output[:,:,4] = x_[:,:,16]
        return output.to(self.precision).to(self.my_device)


        


    def forward(self, image):
        x = self.first_layer(image)
        mu_z = self.linear_celu_stack_mean(x).flatten(1)
        if self.chla_centered == True:
            mu_z += torch.column_stack((x[:,0,0],x[:,0,0],x[:,0,0])) # mean_z = NN1 + epsilon_NN2
            
        #mu_z = self.LogSigmoid(mu_z - 1.3) + 1.3 #analitical solution to clap the value to 1.3 (no mean densities greater than 20 g/m3)
        ###the output of linear_celu_stack_cov_z is the Cholesky decomposition of the cov matrix. 
        Cholesky_z = torch.tril(self.linear_celu_stack_cov(x).flatten(1).reshape((x.shape[0],3,3)))/10


        epsilon = torch.randn(torch.Size([x.shape[0],1,3]),generator=torch.Generator().manual_seed(0)).to(self.my_device)

        z_hat = mu_z + torch.transpose(Cholesky_z@torch.transpose(epsilon,dim0=1,dim1=2),dim0=1,dim1=2).flatten(1) #transforming to do matmul in the correct dimention, then going back to normal.
        z_hat_inter = z_hat
        z_hat = (z_hat * self.y_mul[0] + self.y_add[0]).unsqueeze(1)
        image = self.rearange_RRS(image)
        
        
        rrs_ = self.Forward_Model(image,parameters = z_hat,constant = self.constant)
        

        rrs_ = (rrs_ - self.x_add[:5])/self.x_mul[:5]
 
        kd_ = self.kd(9.,image[:,:,0],image[:,:,1],image[:,:,2],image[:,:,3],image[:,:,4],torch.exp(z_hat[:,:,0]),torch.exp(z_hat[:,:,1]),torch.exp(z_hat[:,:,2]),self.Forward_Model.perturbation_factors,constant = self.constant)

        kd_ = (kd_  - self.y_add[1:6])/self.y_mul[1:6]

        bbp_ = self.bbp(image[:,:,0],image[:,:,1],image[:,:,2],image[:,:,3],image[:,:,4],torch.exp(z_hat[:,:,0]),torch.exp(z_hat[:,:,1]),torch.exp(z_hat[:,:,2]),self.Forward_Model.perturbation_factors,constant = self.constant)[:,[1,2,4]]

        bbp_ = (bbp_ - self.y_add[6:9])/self.y_mul[6:9]

        cov_z = torch.transpose(Cholesky_z,dim0=1,dim1=2) @ Cholesky_z

        return z_hat_inter,cov_z,mu_z,kd_,bbp_,rrs_

class composed_loss_function(nn.Module):

    def __init__(self,precision = torch.float32,my_device = 'cpu',rrs_mul=torch.ones(5),chla_mul=torch.ones(1),kd_mul=torch.ones(5),bbp_mul=torch.ones(3),dk_alpha=0.0001):
        super(composed_loss_function, self).__init__()
        self.dk_alpha=dk_alpha
        self.precision = precision
        self.s_e = torch.diag(torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4])**(2)).to(my_device)
        self.my_device = my_device
        
        s_a = (torch.eye(3)*4.9) #s_a is different to avoid big values of chla

        self.s_a = (chla_mul**(-2) * s_a).to(my_device)

        self.s_a_inv = s_a.inverse().to(my_device)

        rrs_mul = rrs_mul.to(my_device)
        
        
        rrs_cov = torch.diag(rrs_mul**(-1)).T @ (self.s_e @ torch.diag(rrs_mul**(-1))) # s_e is the covariance matriz of rrs before normalization
        
        self.rrs_cov_inv = rrs_cov.inverse().to(torch.float32).to(my_device)

        Y_cov = torch.empty(9)
        Y_cov[0] = chla_mul
        Y_cov[1:6] = kd_mul
        Y_cov[6:] = bbp_mul


        Y_cov = torch.diag(Y_cov**(-1)).T @ (torch.eye(9) @ torch.diag(Y_cov**(-1)))
        self.Y_cov_inv = Y_cov.inverse().to(torch.float32).to(my_device)


    def forward(self,pred_,Y_obs,rrs,rrs_pred,nan_array,cov_z=None,mu_z=None,parameters=[1]):

        rrs_error = torch.trace(   (rrs - rrs_pred) @ ( self.rrs_cov_inv @ (rrs - rrs_pred ).T ) )/(5*pred_.shape[0])
        lens = torch.tensor([len(element[~element.isnan()])  for element in nan_array]).to(self.precision).to(self.my_device)
        obs_error = torch.trace(   ((pred_ - Y_obs) @ ( self.Y_cov_inv @ (pred_ - Y_obs ).T ))/lens )/pred_.shape[0]

        cov_z_inv = inv(cov_z)
        DK = 0.5* torch.sum(torch.log(torch.linalg.det(cov_z)) - torch.log(torch.linalg.det(self.s_a))  + torch.vmap(torch.trace)(cov_z_inv @ self.s_a)   )/pred_.shape[0]\
            + 0.5* torch.sum(  (mu_z.unsqueeze(1) -0.6447) @ cov_z_inv @ torch.transpose((mu_z.unsqueeze(1) -0.6447),dim0=1,dim1=2)) /(3*pred_.shape[0]) #(0 - nn_model.add)/nn_model.mul = 0.6447
        
        l2_norm =  (( parameters - 1 )**2).mean()
        error = rrs_error +  obs_error + DK*self.dk_alpha + l2_norm
            
        return (error).to(self.my_device)



def mask_nans(Y,kd_pred,bbp_pred,chla_pred,my_device = 'cpu'):

    Y_is_nan = torch.isnan(Y)
    Y_masked = torch.masked_fill(Y,Y_is_nan,0)[:,0,:]
    
    pred_masked = torch.zeros(Y_masked.shape[0],Y_masked.shape[1])
    pred_masked[:,1:6] = torch.masked_fill(kd_pred,Y_is_nan[:,0,1:6],0)
    pred_masked[:,6:] = torch.masked_fill(bbp_pred,Y_is_nan[:,0,6:9],0)
    pred_masked[:,0] = torch.masked_fill(chla_pred[:,0],Y_is_nan[:,0,0],0)
    return Y_masked.to(my_device),pred_masked.to(my_device)

def train_one_epoch(epoch_index,training_dataloader,loss_fn,optimizer,model,dates=None,num_samples=1,my_device = 'cpu'):
       
    def one_sample(data):
        inputs,labels_nan = data
        z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model(inputs)
        Y_masked, pred_masked = mask_nans(labels_nan,kd_hat,bbp_hat,z_hat,my_device = my_device)
        # Compute the loss and its gradients

        return loss_fn(pred_masked,Y_masked,inputs[:,0,:5],rrs_hat,labels_nan[:,0,:],cov_z,mu_z,list(model.parameters())[-1])

    def one_loop(data):
        # Every data instance is an input + label pair
        optimizer.zero_grad()
        loss = sum(map(one_sample,[data]*num_samples))
        loss /= num_samples
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        return loss.item()
    
    list_data = list(iter(training_dataloader))
    return np.mean(list(map(one_loop,list_data)))

def validation_loop(epoch_index,validation_dataloader,loss_fn,optimizer,model,my_device = 'cpu'):   
    running_vloss = 0.
    def one_loop(vdata):
        vinputs, vlabels_nan = vdata
        z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model(vinputs)
        Y_masked, pred_masked = mask_nans(vlabels_nan,kd_hat,bbp_hat,z_hat,my_device = my_device)

        # Compute the loss and its gradients

        vloss = loss_fn(pred_masked,Y_masked,vinputs[:,0,:5],rrs_hat,vlabels_nan[:,0,:],cov_z,mu_z,list(model.parameters())[-1])
        return vloss.item()
    with torch.no_grad():
        list_data = list(iter(validation_dataloader))
        running_vloss = np.mean(list(map(one_loop,list_data)))
        
    return running_vloss

def train_cifar(config,data_dir = None):
    time_init = time.time()

    my_device = 'cpu'


    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    
    train_data = rdm.customTensorData(data_path=data_dir,which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')
 
    batch_size = int(config['batch_size'])
    number_hiden_layers_mean = config['number_hiden_layers_mean']
    dim_hiden_layers_mean = config['dim_hiden_layers_mean']
    dim_last_hiden_layer_mean = config['dim_last_hiden_layer_mean']
    alpha_mean = config['alpha_mean']
    number_hiden_layers_cov = config['number_hiden_layers_cov']
    dim_hiden_layers_cov = config['dim_hiden_layers_cov']
    dim_last_hiden_layer_cov = config['dim_last_hiden_layer_cov']
    alpha_cov = config['alpha_cov']
    lr = config['lr']
    betas1 = config['betas1'] 
    betas2 = config['betas2']
    dk_alpha = config['dk_alpha']

    
    mean_validation_loss = 0.
    mean_train_loss = 0.

    for i in range(2):

        train_d,validation = random_split(train_data,[0.95,0.05],generator = torch.Generator().manual_seed(i))
        training_dataloader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation,batch_size = batch_size,shuffle=True)

        model = NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                                 dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                               dim_last_hiden_layer_cov=dim_last_hiden_layer_cov,number_hiden_layers_cov = number_hiden_layers_cov,\
                                 dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,x_mul=train_data.x_mul,x_add=train_data.x_add,\
                                 y_mul=train_data.y_mul,y_add=train_data.y_add,constant = constant,model_dir = '/Users/carlos/Documents/OGS_one_d_model/VAE_model',my_device = my_device)
        
        list(iter(model.parameters()))[-1].requires_grad = False

        loss_function = composed_loss_function(rrs_mul=train_data.x_mul[:5],chla_mul= train_data.y_mul[0],kd_mul= train_data.y_mul[1:6],bbp_mul=train_data.y_mul[6:],dk_alpha=dk_alpha,my_device = my_device)
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
        for epoch in range(start_epoch,50):
            train_loss.append(train_one_epoch(epoch,training_dataloader,loss_function,optimizer,model,my_device = my_device))
            validation_loss.append(validation_loop(epoch,validation_dataloader,loss_function,optimizer,model,my_device = my_device))
            
            
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
            {"loss_validation": mean_validation_loss/2,'loss_train':mean_train_loss/2},
            checkpoint=checkpoint,
        )
    print('Finishe Training {} seconds'.format(time.time() - time_init))

def explore_hyperparameters():
    
    data_dir = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
   
    torch.manual_seed(0)
    
    config_space = CS.ConfigurationSpace({
        "batch_size":[20,25,30],
        
        "number_hiden_layers_mean":list(np.arange(2,5)),
        "dim_hiden_layers_mean":list(np.arange(16,22)),
        "dim_last_hiden_layer_mean":list(np.arange(16,22)) ,
        "alpha_mean":CS.Float("alpha_mean",bounds=(0.5, 3),distribution=CS.Normal(1.3996294280783, 0.1)),

        "number_hiden_layers_cov":list(np.arange(2,5)),
        "dim_hiden_layers_cov":list(np.arange(13,17)),
        "dim_last_hiden_layer_cov":list(np.arange(11,15)) ,
        "alpha_cov":CS.Float("alpha_cov",bounds=(0.5, 3),distribution=CS.Normal(0.7429518278001, 0.1)),
        
        "betas1":CS.Float("betas1",bounds=(0.5, 0.99),distribution=CS.Normal(0.6952126048336  , 0.1)),
        "betas2":CS.Float("betas2",bounds=(0.5, 0.99),distribution=CS.Normal(0.7070708257291, 0.1)),
        "lr":CS.Float("lr",bounds=(0.0001, 0.01),distribution=CS.Normal(0.0083660712025, 0.01),log=True),

        "dk_alpha":CS.Float("dk_alpha",bounds=(0,1),distribution = CS.Normal(0.0001,0.01))
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
            name="bohb_minimization_part2", stop={"training_iteration": max_iterations}
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
    experiment_path = '/Users/carlos/ray_results/bohb_minimization_part2'
    data_dir = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
        
    restored_tuner = tune.Tuner.restore(experiment_path, trainable=partial(train_cifar, data_dir=data_dir))
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result("loss_validation","min")
 
    batch_size = int(best_result.config['batch_size'])
    number_hiden_layers_mean = best_result.config['number_hiden_layers_mean']
    dim_hiden_layers_mean = best_result.config['dim_hiden_layers_mean']
    dim_last_hiden_layer_mean = best_result.config['dim_last_hiden_layer_mean']
    alpha_mean = best_result.config['alpha_mean']
    number_hiden_layers_cov = best_result.config['number_hiden_layers_cov']
    dim_hiden_layers_cov = best_result.config['dim_hiden_layers_cov']
    dim_last_hiden_layer_cov = best_result.config['dim_last_hiden_layer_cov']
    alpha_cov = best_result.config['alpha_cov']
    lr = best_result.config['lr']
    betas1 = best_result.config['betas1'] 
    betas2 = best_result.config['betas2']
    dk_alpha = best_result.config['dk_alpha']

    my_device = 'cpu'

    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    train_data = rdm.customTensorData(data_path=data_dir,which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                                 dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                               dim_last_hiden_layer_cov=dim_last_hiden_layer_cov,number_hiden_layers_cov = number_hiden_layers_cov,\
                                 dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,x_mul=train_data.x_mul,x_add=train_data.x_add,\
                                 y_mul=train_data.y_mul,y_add=train_data.y_add,constant = constant,model_dir = '/Users/carlos/Documents/OGS_one_d_model/VAE_model',my_device = my_device).to(my_device)

    
    loss_function = composed_loss_function(rrs_mul=train_data.x_mul[:5],chla_mul= train_data.y_mul[0],kd_mul= train_data.y_mul[1:6],bbp_mul=train_data.y_mul[6:],dk_alpha=dk_alpha,my_device = my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas1,betas2),maximize=False)
    
    
    validation_loss = []
    train_loss = []

    perturbation_factors_history = np.empty((1001,14))
    perturbation_factors_history[0] = list(iter(model.parameters()))[-1].clone().detach().cpu()
    

    
    list(iter(model.parameters()))[-1].requires_grad = True

    def one_epoch(epoch):
        init_time = time.time()
        train_loss.append(train_one_epoch(epoch,train_dataloader,loss_function,optimizer,model,dates = train_data.dates,num_samples=5,my_device = my_device))
        
        if epoch % 10 == 0:
            print('epoch',epoch,'done in',time.time() - init_time,'seconds','loss:',train_loss[-1])
            torch.save(model.state_dict(), data_dir+'/../VAE_model/model_second_part_save_epoch_'+str(epoch)+'.pt')
        #if epoch == 499:
        #    list(iter(model.parameters()))[-1].requires_grad = True
        #if epoch >= 500:
        perturbation_factors_history[epoch+1] = list(iter(model.parameters()))[-1].clone().detach()
            
        
    list(map(one_epoch,range(1000)))
        
        
    torch.save(model.state_dict(), data_dir+'/../VAE_model/model_second_part_chla_centered.pt')
    np.save(data_dir+'/../plot_data/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy',perturbation_factors_history)
    print('perturbation_factors_history saved in',data_dir+'/../plot_data/perturbation_factors')
    



def save_var_uncertainties(Forward_Model, X, chla_hat_mean, covariance_matrix,rrs_hat, constant = None, save_path ='/Users/carlos/Documents/OGS_one_d_model/VAE_model/results_VAE_VAEparam',dates = [] ):
    parameters_eval = chla_hat_mean.unsqueeze(1)
    evaluate_model = fm.evaluate_model_class(Forward_Model,X,constant=constant)
        
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
               
    kd_delta = fm.error_propagation(kd_derivative,covariance_matrix)
    kd_hat[:,::2] = kd_values.clone().detach()
    kd_hat[:,1::2] = torch.sqrt(kd_delta).clone().detach()
    bbp_values = evaluate_model.bbp_der(parameters_eval)

    bbp_derivative = torch.empty((len(parameters_eval),3,3))
    bbp_derivative_ = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))
    for i in range(len(parameters_eval)):
        bbp_derivative[i] = torch.reshape(bbp_derivative_[i,:,i,:,:],(3,3))
        
    bbp_delta = fm.error_propagation(bbp_derivative,covariance_matrix)
    bbp_hat[:,::2] = bbp_values.clone().detach()
    bbp_hat[:,1::2] = torch.sqrt(bbp_delta).clone().detach()

    rrs_hat = rrs_hat.clone().detach() 

    np.save(save_path + '/X_hat.npy',X_hat)
    np.save(save_path + '/RRS_hat.npy',rrs_hat)
    np.save(save_path + '/kd_hat.npy',kd_hat)
    np.save(save_path + '/bbp_hat.npy',bbp_hat)
    np.save(save_path + '/dates.npy',dates)
    
if __name__ == '__main__':



    #explore_hyperparameters()

    #save_cvae_first_part()

    
    data_dir = HOME_PATH + '/npy_data'
        
    #restored_tuner = tune.Tuner.restore(experiment_path, trainable=partial(train_cifar, data_dir=data_dir))
    #result_grid = restored_tuner.get_results()
    #best_result = result_grid.get_best_result("loss_validation","min")

        
    #torch.save(best_result.config,'/Users/carlos/Documents/OGS_one_d_model/VAE_model/model_second_part_final_config.pt')
    best_result_config = torch.load(HOME_PATH + '/VAE_model/model_second_part_final_config.pt')

    batch_size = int(best_result_config['batch_size'])
    number_hiden_layers_mean = best_result_config['number_hiden_layers_mean']
    dim_hiden_layers_mean = best_result_config['dim_hiden_layers_mean']
    dim_last_hiden_layer_mean = best_result_config['dim_last_hiden_layer_mean']
    alpha_mean = best_result_config['alpha_mean']
    number_hiden_layers_cov = best_result_config['number_hiden_layers_cov']
    dim_hiden_layers_cov = best_result_config['dim_hiden_layers_cov']
    dim_last_hiden_layer_cov = best_result_config['dim_last_hiden_layer_cov']
    alpha_cov = best_result_config['alpha_cov']
    lr = best_result_config['lr']
    betas1 = best_result_config['betas1'] 
    betas2 = best_result_config['betas2']
    dk_alpha = best_result_config['dk_alpha']


    my_device = 'cpu'

    constant = rdm.read_constants(file1=HOME_PATH + '/cte_lambda.csv',file2=HOME_PATH+'/cst.csv',my_device = my_device)
    data = rdm.customTensorData(data_path=data_dir,which='all',per_day = False,randomice=False,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')

    dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)

    model = NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                           dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                           number_hiden_layers_cov = number_hiden_layers_cov,\
                           dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,dim_last_hiden_layer_cov = dim_last_hiden_layer_cov,x_mul=data.x_mul,x_add=data.x_add,\
                           y_mul=data.y_mul,y_add=data.y_add,constant = constant,model_dir = HOME_PATH + '/VAE_model').to(my_device)

    model.load_state_dict(torch.load(HOME_PATH + '/VAE_model/model_second_part_chla_centered.pt'))
    X,Y = next(iter(dataloader))
    
    z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model(X)

    plt.plot(z_hat[:,0].clone().detach())
    plt.show()

    mu_z = mu_z* data.y_mul[0] + data.y_add[0]
    cov_z = torch.diag(data.y_mul[0].expand(3)).T @ cov_z @ torch.diag(data.y_mul[0].expand(3)) 
    kd_hat = kd_hat * data.y_mul[1:6] + data.y_add[1:6]
    bbp_hat = bbp_hat * data.y_mul[6:] + data.y_add[6:]
    rrs_hat = rrs_hat * data.x_mul[:5] + data.x_add[:5]
    X = model.rearange_RRS(X)
        
    save_var_uncertainties(model.Forward_Model,X,mu_z,cov_z,rrs_hat,constant=constant,dates = data.dates,save_path =HOME_PATH + '/VAE_model/results_VAE_VAEparam_chla')
    
    


    

    
    

