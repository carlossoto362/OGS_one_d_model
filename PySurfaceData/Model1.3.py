#!/usr/bin/env python

"""
Functions for learning the constants for the inversion problem.

As part of the National Institute of Oceanography, and Applied Geophysics, I'm working on an inversion problem. A detailed description can be found at
https://github.com/carlossoto362/firstModelOGS.
The inversion model contains the functions required to reed the satellite data and process it, in order to obtain the constituents: (chl-a, CDOM, NAP), 
using the first introduced model.

In addition, some of the constants are now learnable parameters, and there is a function that uses the historical data to learn the parameters. 
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



class customTensorData():
    def __init__(self,data_path='./npy_data',transform=None,target_transform=None,train_percentage = 0.9,from_where = 'left',randomice=False,specific_columns=None,which='train',seed=None,per_day=True):

        x_data = np.load(data_path + '/x_data_all.npy',allow_pickle=False)
        self.dates = x_data[:,-1]
        self.init = x_data[:,22:25]
        self.x_data = np.delete(x_data,[22,23,24,25],axis=1)
        

        self.x_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555','Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','lambda_412','lambda_442',\
                               'lambda_490','lambda_510','lambda_555','zenith','PAR']
        self.per_day = per_day
        if self.per_day == True:
            self.y_data = self.x_data[:,:5]
            self.y_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555']
            self.x_data = np.delete(self.x_data,[0,1,2,3,4],axis=1)
        else:
            y_data = np.load(data_path + '/y_data_all.npy',allow_pickle=False)
            self.y_column_names = ['chla','kd_412','kd_442','kd_490','kd_510','kd_555','bbp_442','bbp_490','bbp_555']

        self.init_column_names = ['chla_init','NAP_init','CDOM_init']
        
        self.date_info = '''date indicating the number of days since the first of january of 2000.'''



        if specific_columns != None:
            self.x_data = self.x_data[specific_columns]
            self.x_column_names = self.x_column_names[specific_columns]

        self.len_data = len(self.dates)
        self.indexes = np.arange(self.len_data)
        if randomice == True:
            if type(seed) != type(None):
                np.random.seed(seed)
            np.random.shuffle(self.indexes)

        if from_where == 'right':
            self.indexes = np.flip(self.indexes)

        self.train_indexes = self.indexes[:int(self.len_data * train_percentage)]
        self.test_indexes = self.indexes[int(self.len_data * train_percentage):]

        self.which = which
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.my_indexes = self.indexes
        if self.which.lower().strip() == 'train' :
            self.my_indexes = self.train_indexes
        elif self.which.lower().strip() == 'test':
            self.my_indexse = self.test_indexes

    def __len__(self):

            return len(self.my_indexes)

    def __getitem__(self, idx):
        
        if self.per_day == True:
            label = torch.empty((5,1))
            label[:,0] = torch.tensor(self.y_data[self.my_indexes][idx])

            image = torch.empty((5,5))
            image[:,0] = torch.tensor(self.x_data[self.my_indexes][idx][:5])
            image[:,1] = torch.tensor(self.x_data[self.my_indexes][idx][5:10])
            image[:,2] = torch.tensor(self.x_data[self.my_indexes][idx][10:15])
            image[:,3] = torch.tensor(self.x_data[self.my_indexes][idx][15])
            image[:,4] = torch.tensor(self.x_data[self.my_indexes][idx][16])
        else:
            label = torch.tensor(self.y_data[self.my_indexes][idx])

            image = torch.empty((5,6))
            image[:,0] = torch.tensor(self.x_data[self.my_indexes][idx][:5])
            image[:,1] = torch.tensor(self.x_data[self.my_indexes][idx][5:10])
            image[:,3] = torch.tensor(self.x_data[self.my_indexes][idx][10:15])
            image[:,4] = torch.tensor(self.x_data[self.my_indexes][idx][15:20])
            image[:,5] = torch.tensor(self.x_data[self.my_indexes][idx][20])
            image[:,6] = torch.tensor(self.x_data[self.my_indexes][idx][21])
          
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




def read_constants(file1='./cte_lambda.csv',file2='./cst.csv'):
    """
    function that reads the constants stored in file1 and file2. 
    file1 has the constants that are dependent on lambda, is a csv with the columns
    lambda, absortion_w, scattering_w, backscattering_w, absortion_PH, scattering_PH, backscattering_PH.
    file2 has the constants that are independent of lambda, is a csv with the columns
    name,values.

    read_constants(file1,file2) returns a dictionary with all the constants. To access the absortion_w for examplea, write 
    constant = read_constants(file1,file2)['absortion_w']['412.5'].
    """

    cts_lambda = pd.read_csv(file1)
    constant = {}
    for key in cts_lambda.keys()[1:]:
        constant[key] = {}
        for i in range(len(cts_lambda['lambda'])):
            constant[key][str(cts_lambda['lambda'].iloc[i])] = cts_lambda[key].iloc[i]
        cts = pd.read_csv(file2)
        
    for i in range(len(cts['name'])):
        constant[cts['name'].iloc[i]] = cts['value'].iloc[i]
    return constant

constant = read_constants(file1='cte_lambda.csv',file2='cst.csv')
lambdas = np.array([412.5,442.5,490,510,555]).astype(float)

linear_regression=stats.linregress(lambdas,[constant['scattering_PH'][str(lamb)] for lamb in lambdas])
linear_regression_slope = linear_regression.slope
linear_regression_intercept = linear_regression.intercept




################Functions for the absortion coefitient####################
def absortion_CDOM(lambda_,perturbation_factors,tensor = True):
    """
    Function that returns the mass-specific absorption coefficient of CDOM, function dependent of the wavelength lambda. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
        return constant['dCDOM']*np.exp(-(constant['sCDOM'] * perturbation_factors['mul_factor_s_cdom'])*(lambda_ - 450.))
    else:
        return constant['dCDOM']*torch.exp(-(constant['sCDOM'] * perturbation_factors['mul_factor_s_cdom'])*(torch.tensor(lambda_) - 450.))

def absortion_NAP(lambda_,tensor = True):
    """
    Mass specific absorption coefficient of NAP.
    See Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
    	return constant['dNAP']*np.exp(-constant['sNAP']*(lambda_ - 440.))
    else:
    	return constant['dNAP']*torch.exp(-constant['sNAP']*(torch.tensor(lambda_) - 440.))

def absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=True):
    """
    Total absortion coeffitient.
    aW,λ (values used from Pope and Fry, 1997), aP H,λ (values averaged and interpolated from
    Alvarez et al., 2022).
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
        return torch.tensor(list(constant['absortion_w'].values())).reshape((5,1)) + (torch.tensor(list(constant['absortion_PH'].values())).reshape((5,1)) * perturbation_factors['mul_factor_a_ph'])*chla + \
        (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor)* perturbation_factors['mul_factor_a_cdom'])*CDOM + absortion_NAP(lambda_,tensor=tensor)*NAP
    else:
        return np.array(list(constant['absortion_w'].values())).reshape((5,1)) + (list(constant['absortion_PH'].values()) * perturbation_factors['mul_factor_a_ph'])*chla + \
        (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor)* perturbation_factors['mul_factor_a_cdom'])*CDOM + absortion_NAP(lambda_,tensor=tensor)*NAP


##############Functions for the scattering coefitient########################
def Carbon(chla,PAR, perturbation_factors,tensor=True):
    """
    defined from the carbon to Chl-a ratio. 
    theta_o, sigma, beta, and theta_min constants (equation and values computed from Cloern et al., 1995), and PAR
    the Photosynthetically available radiation, obtained from the OASIM model, see Lazzari et al., 2021.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    nominator = chla
    beta =  constant['beta'] * perturbation_factors['mul_factor_beta']
    sigma = constant['sigma'] * perturbation_factors['mul_factor_sigma']
    exponent = -(PAR - beta)/sigma
    if tensor == False:
        denominator = (constant['Theta_o']* perturbation_factors['mul_factor_theta_o']) * ( np.exp(exponent)/(1+np.exp(exponent)) ) + \
        (constant['Theta_min'] * perturbation_factors['mul_factor_theta_min'])
    else:
        denominator = (constant['Theta_o']* perturbation_factors['mul_factor_theta_o']) * ( torch.exp(exponent)/(1+torch.exp(exponent)) ) + \
        (constant['Theta_min'] * perturbation_factors['mul_factor_theta_min'])
    return nominator/denominator

def scattering_ph(lambda_,perturbation_factors,tensor = True):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    
    return (linear_regression_slope * perturbation_factors['mul_tangent_b_ph']) *\
        lambda_ + linear_regression_intercept * perturbation_factors['mul_intercept_b_ph']

def scattering_NAP(lambda_,tensor=True):
    """
    NAP mass-specific scattering coefficient.
    eNAP and fNAP constants (equation and values used from Gallegos et al., 2011)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return constant['eNAP']*(550./lambda_)**constant['fNAP']

def scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    Total scattering coefficient.
    bW,λ (values interpolated from Smith and Baker, 1981,), bP H,λ (values used
    from Dutkiewicz et al., 2015)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
        return torch.tensor(list(constant['scattering_w'].values())).reshape((5,1)) + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
        scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        return np.array(list(constant['scattering_w'].values())).reshape((5,1)) + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
        scattering_NAP(lambda_,tensor=tensor) * NAP

#################Functions for the backscattering coefitient#############

def backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
        return torch.tensor(list(constant['backscattering_w'].values())).reshape((5,1)) + perturbation_factors['mul_factor_backscattering_ph']*torch.tensor(list(constant['backscattering_PH'].values())).reshape((5,1)) * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        return np.array(list(constant['backscattering_w'].values())).reshape((5,1)) + perturbation_factors['mul_factor_backscattering_ph']*np.array(list(constant['backscattering_PH'].values())) * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP



###############Functions for the end solution of the equations###########
#The final result is written in terms of these functions, see ...

def c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True): 
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/torch.cos(torch.tensor(zenith)*3.1416/180)
    else:
    	return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/np.cos(zenith*3.1416/180)

def F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/\
        torch.cos(torch.tensor(zenith)*3.1416/180.)
    else:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/\
        np.cos(zenith*3.1416/180.)

def B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)/torch.cos(torch.tensor(zenith)*3.1416/180) 
    else:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)/np.cos(zenith*3.1416/180)

def C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) )/\
        constant['vs']

def B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/constant['vu']

def B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/constant['vs']

def C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor) + constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor))/\
        constant['vu']

def D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (0.5) * (C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + \
                    ((C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor))**2 -\
                     4. * B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) )**(0.5))

def x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)
    nominator = -(C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) *\
        F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor) -\
        B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor)

    return nominator/denominator

def y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)
    nominator = (-B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor) * F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor) ) +\
        (-C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)) *\
        B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor)

    return nominator/denominator

def C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return E_dif_o - x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * E_dir_o

def r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor)/D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)

def k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) - C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)

def E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    This is the analytical solution of the bio-chemical model. (work not published.)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor==False:
        return E_dir_o*np.exp(-z*c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor))
    else:
        return E_dir_o*torch.exp(-z*c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors))
        

def E_u(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True):
    """
    This is the analytical solution of the bio-chemical model. (work not published.)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)*\
        np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)*z)+\
        y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors)*\
        torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors)*z)+\
        y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors)

def E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    This is the analytical solution of the bio-chemical model. (work not published.)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    if tensor == False:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) *\
        np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)*z)+\
        x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors) *\
        torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors)*z)+\
        x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors)
        

def bbp(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    Particulate backscattering at depht z
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
        return torch.tensor(list(constant['backscattering_PH'].values())).reshape((5,1)) * perturbation_factors['mul_factor_backscattering_ph'] * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        return np.array(list(constant['backscattering_PH'].values())).reshape((5,1)) * perturbation_factors['mul_factor_backscattering_ph'] * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP

def kd(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True):
    """
    Atenuation Factor
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor==False:
        return (z**-1)*np.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)))
    else:

        return (z**-1)*torch.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors)))

##########################from the bio-optical model to RRS(Remote Sensing Reflectance)##############################
#defining Rrs
#Q=5.33*np.exp(-0.45*np.sin(np.pi/180.*(90.0-Zenith)))

def Q_rs(zenith,perturbation_factors,tensor=True):
    """
    Empirical result for the Radiance distribution function, 
    equation from Aas and Højerslev, 1999, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor==True:
        return (5.33 * perturbation_factors['mul_factor_q_a'])*torch.exp(-(0.45 * perturbation_factors['mul_factor_q_b'])*torch.sin((3.1416/180.0)*(90.0-torch.tensor(zenith))))
    else:
        return  (5.33 * perturbation_factors['mul_factor_q_a'])*np.exp(-(0.45 * perturbation_factors['mul_factor_q_b'])*np.sin((3.1416/180.0)*(90.0-zenith)))

def Rrs_minus(Rrs,tensor=True):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return Rrs/(constant['T']+constant['gammaQ']*Rrs)

def Rrs_plus(Rrs,tensor=True):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return Rrs*constant['T']/(1-constant['gammaQ']*Rrs)

def Rrs_MODEL(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True):
    """
    Remote Sensing Reflectance.
    Aas and Højerslev, 1999.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    Rrs = E_u(0,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor)  /  (   Q_rs(zenith,perturbation_factors,tensor=tensor)*(E_dir_o + E_dif_o)   )
    return Rrs_plus( Rrs ,tensor = tensor)



class MODEL(nn.Module):
    """
    Bio-Optical model plus corrections, in order to have the Remote Sensing Reflectance, in terms of the inversion problem. 
    MODEL(x) returns a tensor, with each component being the Remote Sensing Reflectance for each given wavelength. 
    if the data has 5 rows, each with a different wavelength, RRS will return a vector with 5 components.  RRS has tree parameters, 
    self.chla is the chlorophil-a, self.NAP the Non Algal Particles, and self.CDOM the Colored Dissolved Organic Mather. 
    According to the invention problem, we want to estimate them by making these three parameters have two constraints,
    follow the equations of the bio-optical model, plus, making the RRS as close as possible to the value
    measured by the satellite.
    
    """
    def __init__(self):
        super().__init__()
        
        self.chparam = nn.Parameter(torch.ones((3), dtype=torch.float32), requires_grad=True)
        
        self.chla = self.chparam[0]
        self.NAP = self.chparam[1]
        self.CDOM = self.chparam[2]

        self.perturbation_factors = {
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

    def forward(self,x_data):
        """
        x_data: pandas dataframe with columns [E_dif,E_dir,lambda,zenith,PAR].
        """

        #Rrs[i,] = Rrs_MODEL(x_data['E_dif'].iloc[i],x_data['E_dir'].iloc[i],x_data['lambda'].iloc[i],\
        #x_data['zenith'].iloc[i],x_data['PAR'].iloc[i],self.chla,self.NAP,self.CDOM,perturbation_factors)
        Rrs = Rrs_MODEL(x_data[:,0],x_data[:,1],x_data[:,2],\
                        x_data[:,3],x_data[:,4],self.chla,self.NAP,self.CDOM,self.perturbation_factors)
        return Rrs


def train_loop(data_i,model,loss_fn,optimizer,N,kind='all'):
    """
    The train loop evaluates the Remote Sensing Reflectance RRS for each wavelength>>>pred=model(data_i), evaluates the loss function
    >>>loss=loss_fn(pred,y), force the value of the parameters (chla,NAP,CDOM) to be positive, evaluates the gradient of RRS with respect
    to the parameters, >>>loss.backward(), modifies the value of the parameters according to the optimizer criterium, >>>optimizer.step(),
    sets the gradient of RRS to cero, and prints the loss for a given number of iterations. This procedure is performed N times or untyl a treshold is ashieved. 
    After N iterations, it returns two lists with the evolution of the loss function and the last evaluation of the model. 
    """
    
    ls_val=[]
    past_pred=torch.empty((N,3))

    time_init = time.time()

    criterium = 1
    criterium_2 = 0
    i=0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    X = data_i[0][0]
    Y = data_i[1][0]
    s_a = (loss_fn.s_a)
    s_e = (loss_fn.s_e)
    s_e_inverse = (loss_fn.s_e_inverse)
    s_a_inverse = (loss_fn.s_a_inverse)
    
    while (((criterium >1e-12 ) & (i<N)) or ((criterium_2 < 100)&(i<N))):
        
        pred = model(X)
        loss = loss_fn(Y,pred,model.state_dict()['chparam'])
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for index_,p in enumerate(model.parameters()):
                p.data.clamp_(0)
                
        ls_val.append(loss.item())
        past_pred[i] = model.state_dict()['chparam']
        
        if i != 0:
            criterium = ls_val[-2] - ls_val[-1]
        if criterium <=1e-12:
            criterium_2+=1
        i+=1
        scheduler.step(loss)
    last_i = i
    if kind == 'all':
        class evaluate_model_class():
            def __init__(self,axis=None):
                self.axis = axis
            
            def model_der(self,parameters,X_=X):
                return model(X_,parameters = parameters,axis = self.axis)
        
            def kd_der(self,parameters):
                if self.axis == None:
                    kd_values = torch.empty(5)
                    for i in range(5):
                        kd_values[i] = kd(0.1,X[5+i],X[10+i],lambdas[i],\
                                      X[15],X[16],*parameters,model.perturbation_factors)
                    return kd_values
                else:
                    return kd(0.1,X[5+self.axis],X[10+self.axis],lambdas[self.axis],\
                                  X[15],X[16],*parameters,model.perturbation_factors)

            def bbp_der(self,parameters):
                if self.axis == None:
                    bbp_values = torch.empty(5)
                    for i in range(5):
                        bbp_values[i] = bbp(X[5+i],X[10+i],lambdas[i],\
                                        X[15],X[16],*parameters,model.perturbation_factors)
                    return bbp_values
                else:
                    return bbp(X[5+self.axis],X[10+self.axis],lambdas[self.axis],\
                                    X[15],X[16],*parameters,model.perturbation_factors)

        parameters_eval = list(model.parameters())[0].clone().detach()
        evaluate_model = evaluate_model_class()
    
        K_x = torch.autograd.functional.jacobian(evaluate_model.model_der,inputs=(parameters_eval))
        S_hat = torch.inverse( K_x.T @ ( s_e_inverse @ K_x ) + s_a_inverse  )
        G_hat = torch.inverse( s_a_inverse + K_x.T @ ( s_e_inverse @ K_x  )  ) @ (  K_x.T @ s_e_inverse  )
        A_hat = G_hat@K_x

        def error_propagation(df,sigma,d_two_f=None):
            error = 0
            if d_two_f == None:
                error = torch.empty(df.size()[0])
                for i in range(df.size()[0]):
                    error[i] = torch.dot( df[i]  , sigma @ df[i]    )
            else:
                error = torch.empty(df.size()[0])
                for i in range(df.size()[0]):
                    error[i] = torch.dot( df[i]  , sigma @ df[i]    ) + torch.trace( d_two_f[i].T @ sigma )**2 + torch.trace( (d_two_f[i].T @ sigma) @ (d_two_f[i] @ sigma.T)  ) + torch.trace( (d_two_f[i] @ sigma)@(d_two_f[i] @ sigma.T)  )
            return error

        lambdas = [555.,510.,490.,442.5,412.5]
        kd_hat = torch.empty(10)
        bbp_hat = torch.empty(6)
        bbp_index = 0


  
        kd_values = evaluate_model.kd_der(parameters_eval)

        kd_derivative = torch.autograd.functional.jacobian(evaluate_model.kd_der,inputs=(parameters_eval))
        kd_two_derivative = torch.empty((5,3,3))
    
        for i in range(5):
            evaluate_model.axis = i
            kd_two_derivative[i] = torch.autograd.functional.hessian(evaluate_model.kd_der,inputs=(parameters_eval))
        
        kd_delta = error_propagation(kd_derivative,S_hat,kd_two_derivative)
        kd_hat[::2] = kd_values
        kd_hat[1::2] = torch.sqrt(kd_delta)
        evaluate_model.axis=None
    
        bbp_values = evaluate_model.bbp_der(parameters_eval)
        #print('value',bbp_values)
        bbp_derivative = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))[[0,2,3]]
        bbp_two_derivative = torch.empty((3,3,3))
        bbp_values = bbp_values[[0,2,3]]
    
        for j,i in enumerate([0,2,3]):
            evaluate_model.axis = i
            bbp_two_derivative[j] = torch.autograd.functional.hessian(evaluate_model.bbp_der,inputs=(parameters_eval))
        print('derivative',bbp_derivative)
        bbp_delta = error_propagation(bbp_derivative,S_hat,bbp_two_derivative)
        #print('delta',torch.sqrt(bbp_delta))
        bbp_hat[::2] = bbp_values
        bbp_hat[1::2] = torch.sqrt(bbp_delta)
        #print('hat',bbp_hat)
        evaluate_model.axis=None
        #checking for linearity
        #Im arguin that
        delta_parameters = torch.tensor([S_hat[0][0].clone().detach(),S_hat[0][1].clone().detach(),S_hat[0][2].clone().detach()])
        def test_linearity():
            new_parameters = list(model.parameters())[0].clone().detach() + delta_parameters
            delta_model = model(X,parameters = new_parameters) - model(X)
            linear_delta_model = torch.sum(K_x * delta_parameters,axis=1)
            print((delta_model-linear_delta_model)/delta_model)

        
        def test_error():
            print('error propagation:',kd_hat[-2:].numpy())
            kd_rand = []
            for i in range(10000):
                parameters =  parameters_eval  + delta_parameters*torch.randn(3)
                kd_rand.append(kd(0.1,X[5+4],X[10+4],lambdas[4],\
                             X[15],X[16],*parameters,model.perturbation_factors))
            ks_test = stats.kstest((kd_rand - np.mean(kd_rand))/np.std(kd_rand),
             stats.norm.cdf)
            n,bins,patches = plt.hist((kd_rand - np.mean(kd_rand))/np.std(kd_rand),bins=30,density=True,label='centered kd distribution')
            gaussian = ((1 / (np.sqrt(2 * np.pi) * 1)) * np.exp(-0.5 * (1 / 1 * (bins - 0.))**2))
            plt.plot(bins,gaussian,'--',alpha=0.5,label='$\eta(0,1)$\nks: {:.5f}\np-value: {:.5f}'.format(*ks_test))
            plt.legend()
            plt.show()
            kd_rand = torch.tensor(kd_rand).numpy()
            print('montecarlo sampling:',np.nanmean(kd_rand),np.nanstd(kd_rand))
    
        output = {'X_hat':past_pred[last_i-1].clone().detach(),'S_hat':S_hat.clone().detach(),'K_hat':K_x.clone().detach(),'G_hat':G_hat.clone().detach(),'A_hat':A_hat.clone().detach(),\
              'Y_hat':pred.clone().detach(),'kd':kd_hat.clone().detach(),'bbp':bbp_hat.clone().detach()}
    
        print("time for training...",time.time() - time_init)
        return ls_val,past_pred[:last_i].clone().detach(),output
    else:
        return past_pred[last_i-1].clone().detach()


class custom_Loss(nn.Module):

    def __init__(self,x_a,s_a,s_e):
        super(custom_Loss, self).__init__()
        self.x_a = x_a
        self.s_a = s_a
        self.s_e = s_e
        self.s_e_inverse = torch.inverse(s_e)
        self.s_a_inverse = torch.inverse(s_a)

    def forward(self,y,f_x,x):
        return  torch.dot((y - f_x),self.s_e_inverse @ (y - f_x )) + torch.dot( (x - self.x_a), self.s_a_inverse @ (x - self.x_a))

if __name__ == '__main__':

    initial_conditions_path = '/Users/carlos/Documents/surface_data_analisis/npy_data/'
    data_path = '/Users/carlos/Documents/surface_data_analisis/npy_data/'
    train_data = customTensorData(data_path='./npy_data',which='train')
    test_data = customTensorData(data_path='./npy_data',which='test')

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    #mps_device = torch.device("mps")
    
    model = MODEL().to('cpu')

    evaluation_time = []

    for i in range(1000):

    
        X,Y = train_data.__getitem__(i)
        init_time = time.time()
    
        model(X)

        evaluation_time.append(time.time() - init_time)

    print(np.mean(evaluation_time))




