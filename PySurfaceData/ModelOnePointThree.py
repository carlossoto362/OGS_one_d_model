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

    def __init__(self,data_path='./npy_data',transform=None,target_transform=None,train_percentage = 0.9,from_where = 'left',randomice=False,specific_columns=None,which='train',seed=None,per_day=True,precision=torch.float32):
        """
        Class used to read the data, x_data is the imput data, y_data is the spected output of the model. It can be the Remote Sensing Reflectance or in-situ messuraments.   
        
        Remote Sensing Reflectance (RRS) from https://data.marine.copernicus.eu/product/OCEANCOLOUR_MED_BGC_L3_MY_009_143/services, values in sr^-1
        Diffracted irradiance in the upper surface of the sea (Edif) from the OASIM model, values in W/m^2.
        Direct irradiance in the upper surface of the sea (Edir) from the OASIM model, values in W/m^2.
        Wave lenghts (lambda), 412.5, 442.5,490,510 and 555, values in nm. 
        Zenith angle (zenith) from the OASIM model, values in degrees.
        Photosynthetic Available Radiation (PAR) from the OASIM, values in W/m^2. 

        the in-situ messuraments are

        Concentration of Chlorophyll-a in the upper layer of the sea (chla), values in mg/m^3
        Downward light attenuation coeffitient (kd), values in m^-1
        Backscattering from phytoplancton and Non Algal Particles (bbp), values in m^-1.

        All data is from the Boussole site. 
        """

        x_data = np.load(data_path + '/x_data_all.npy',allow_pickle=False)
        self.dates = x_data[:,-1]
        self.init = x_data[:,22:25]
        self.x_data = np.delete(x_data,[22,23,24,25],axis=1)
        

        self.x_column_names = ['Edif_412','Edif_442','Edif_490','Edif_510',\
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
        self.precision=precision

    def __len__(self):

            return len(self.my_indexes)

    def __getitem__(self, idx):
        
        if self.per_day == True:
            label = torch.empty((5))
            label[:] = torch.tensor(self.y_data[self.my_indexes][idx])

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
        return image.to(self.precision), label.to(self.precision)




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



###################################################################################################################################################################################################
############################################################################FUNCTIONS NEEDED TO DEFINE THE FORWARD MODEL###########################################################################
###################################################################################################################################################################################################

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

def absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=True,axis=None):
    """
    Total absortion coeffitient.
    aW,λ (values used from Pope and Fry, 1997), aP H,λ (values averaged and interpolated from
    Alvarez et al., 2022).
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if axis == None:
        if tensor == True:
            return torch.tensor(list(constant['absortion_w'].values())) + (torch.tensor(list(constant['absortion_PH'].values())) * perturbation_factors['mul_factor_a_ph'])*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor)* perturbation_factors['mul_factor_a_cdom'])*CDOM + absortion_NAP(lambda_,tensor=tensor)*NAP
        else:
            return np.array(list(constant['absortion_w'].values())) + np.array(list(constant['absortion_PH'].values()) * perturbation_factors['mul_factor_a_ph'])*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor)* perturbation_factors['mul_factor_a_cdom'])*CDOM + absortion_NAP(lambda_,tensor=tensor)*NAP
    else:
        if tensor == True:
            return torch.tensor(list(constant['absortion_w'].values()))[axis] + (torch.tensor(list(constant['absortion_PH'].values()))[axis] * perturbation_factors['mul_factor_a_ph'])*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor)* perturbation_factors['mul_factor_a_cdom'])*CDOM + absortion_NAP(lambda_,tensor=tensor)*NAP
        else:
            return np.array(list(constant['absortion_w'].values()))[axis] + (np.array(list(constant['absortion_PH'].values()))[axis] * perturbation_factors['mul_factor_a_ph'])*chla + \
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

def scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None):
    """
    Total scattering coefficient.
    bW,λ (values interpolated from Smith and Baker, 1981,), bP H,λ (values used
    from Dutkiewicz et al., 2015)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if axis == None:
        if tensor == True:
            return torch.tensor(list(constant['scattering_w'].values())) + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
                scattering_NAP(lambda_,tensor=tensor) * NAP
        else:
            return np.array(list(constant['scattering_w'].values())) + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
                scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        if tensor == True:
            return torch.tensor(list(constant['scattering_w'].values()))[axis] + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
                scattering_NAP(lambda_,tensor=tensor) * NAP
        else:
            return np.array(list(constant['scattering_w'].values()))[axis] + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
                scattering_NAP(lambda_,tensor=tensor) * NAP

#################Functions for the backscattering coefitient#############

def backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if axis == None:
        if tensor == True:
            return torch.tensor(list(constant['backscattering_w'].values())) + perturbation_factors['mul_factor_backscattering_ph']*torch.tensor(list(constant['backscattering_PH'].values())) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
        else:
            return np.array(list(constant['backscattering_w'].values())) + perturbation_factors['mul_factor_backscattering_ph']*np.array(list(constant['backscattering_PH'].values())) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        if tensor == True:
            return torch.tensor(list(constant['backscattering_w'].values()))[axis] + perturbation_factors['mul_factor_backscattering_ph']*torch.tensor(list(constant['backscattering_PH'].values()))[axis] * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
        else:
            return np.array(list(constant['backscattering_w'].values()))[axis] + perturbation_factors['mul_factor_backscattering_ph']*np.array(list(constant['backscattering_PH'].values()))[axis] * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP



###############Functions for the end solution of the equations###########
#The final result is written in terms of these functions, see ...

def c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None): 
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis))/torch.cos(torch.tensor(zenith)*3.1416/180)
    else:
    	return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis))/np.cos(zenith*3.1416/180)

def F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis))/\
        torch.cos(torch.tensor(zenith)*3.1416/180.)
    else:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis))/\
        np.cos(zenith*3.1416/180.)

def B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis)/torch.cos(torch.tensor(zenith)*3.1416/180) 
    else:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis)/np.cos(zenith*3.1416/180)

def C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis) + constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) )/\
        constant['vs']

def B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis))/constant['vu']

def B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis))/constant['vs']

def C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis) + constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis))/\
        constant['vu']

def D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (0.5) * (C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) + \
                    ((C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis))**2 -\
                     4. * B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) )**(0.5))

def x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis)
    nominator = -(C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)) *\
        F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) -\
        B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) * B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis)

    return nominator/denominator

def y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis)
    nominator = (-B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) * F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis) ) +\
        (-C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)) *\
        B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis)

    return nominator/denominator

def C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return E_dif_o - x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) * E_dir_o

def r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis)/D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)

def k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) - C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)


def E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
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
        return E_dir_o*np.exp(-z*c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis))
    else:
        return E_dir_o*torch.exp(-z*c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis))

def E_u(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True,axis=None):
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

        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)*\
                np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)*z)+\
                y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis)*\
                torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis)*z)+\
                y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis)

def E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis = None):
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

        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) *\
                np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)*z)+\
                x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis) *\
                torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis)*z)+\
                x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis)
        

def bbp(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
    """
    Particulate backscattering at depht z
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if axis == None:
        if tensor == True:
            return torch.tensor(list(constant['backscattering_PH'].values())) * perturbation_factors['mul_factor_backscattering_ph'] * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
        else:
            return np.array(list(constant['backscattering_PH'].values())) * perturbation_factors['mul_factor_backscattering_ph'] * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        if tensor == True:
            return torch.tensor(list(constant['backscattering_PH'].values()))[axis] * perturbation_factors['mul_factor_backscattering_ph'] * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
        else:
            return np.array(list(constant['backscattering_PH'].values()))[axis] * perturbation_factors['mul_factor_backscattering_ph'] * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP

def kd(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None):
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
        return (z**-1)*np.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis = axis) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis = axis)))
    else:

        return (z**-1)*torch.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis)))

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

def Rrs_MODEL(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True,axis = None):
    """
    Remote Sensing Reflectance.
    Aas and Højerslev, 1999.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    Rrs = E_u(0,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis = axis)  /  (   Q_rs(zenith,perturbation_factors,tensor=tensor)*(E_dir_o + E_dif_o)   )
    return Rrs_plus( Rrs ,tensor = tensor)


###################################################################################################################################################################################################
#########################################################################FUNCTIONS NEEDED TO FIND THE OPTICAL CONSTITUENTS#########################################################################
###################################################################################################################################################################################################
                
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
    def __init__(self,precision = torch.float32):
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
        self.precision = precision

    def forward(self,x_data,parameters = None, axis = None,perturbation_factors_ = None):
        """
        x_data: pandas dataframe with columns [E_dif,E_dir,lambda,zenith,PAR].
        """
        if type(perturbation_factors_) == type(None):
            perturbations = self.perturbation_factors
        else:
            perturbations = perturbation_factors_
        if type(parameters) == type(None):

            if type(axis) == type(None):
            
                Rrs = Rrs_MODEL(x_data[:,0],x_data[:,1],x_data[:,2],\
                                x_data[:,3],x_data[:,4],self.chla,self.NAP,self.CDOM,perturbations)
            
                return Rrs.to(self.precision)
            else:

                Rrs = Rrs_MODEL(x_data[axis,0],x_data[axis,1],x_data[axis,2],\
                                x_data[axis,3],x_data[axis,4],self.chla,self.NAP,self.CDOM,perturbations)
            
                return Rrs.to(self.precision)

        else:
            if type(axis) == type(None):
                
                Rrs = Rrs_MODEL(x_data[:,0],x_data[:,1],x_data[:,2],\
                                x_data[:,3],x_data[:,4],*parameters,perturbations)
            
                return Rrs.to(self.precision)
            else:
                Rrs = Rrs_MODEL(x_data[axis,0],x_data[axis,1],x_data[axis,2],\
                                x_data[axis,3],x_data[axis,4],*parameters,perturbations)
            
                return Rrs.to(self.precision)
                
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

class evaluate_model_class():
    """
    class to evaluate functions needed to compute the uncerteinty. 
    """
    def __init__(self,model,X,axis=None):
        self.axis = axis
        self.model = model
        self.X = X
        
    def model_der(self,parameters,perturbation_factors_ = None):
        
        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        return self.model(self.X,parameters = parameters,axis = self.axis,perturbation_factors_ = perturbations)
        
    def kd_der(self,parameters,perturbation_factors_ = None):
        
        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        if self.axis == None:
            kd_values = kd(9,self.X[:,0],self.X[:,1],self.X[:,2],\
                           self.X[:,3],self.X[:,4],*parameters,perturbations)
            return kd_values
        else:
            kd_values = kd(9,self.X[self.axis,0],self.X[self.axis,1],self.X[self.axis,2],\
                           self.X[self.axis,3],self.X[self.axis,4],*parameters,perturbations,axis = self.axis)
            return kd_values

    def bbp_der(self,parameters,perturbation_factors_ = None):

        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        if self.axis == None:
            bbp_values = bbp(self.X[:,0],self.X[:,1],self.X[:,2],\
                             self.X[:,3],self.X[:,4],*parameters,perturbations)
            return bbp_values[[1,2,4]]
        else:
            bbp_values = bbp(self.X[self.axis,0],self.X[self.axis,1],self.X[self.axis,2],\
                             self.X[self.axis,3],self.X[self.axis,4],*parameters,perturbations,axis=self.axis)
            return bbp_values
            
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
    
    while (((criterium >1e-13 ) & (i<N)) or ((criterium_2 < 100)&(i<N))):
        
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
        if criterium <=1e-13:
            criterium_2+=1
        i+=1
        scheduler.step(loss)
    last_i = i
    last_rrs = pred

    
    if kind == 'all':
        parameters_eval = list(model.parameters())[0].clone().detach()
        evaluate_model = evaluate_model_class(model=model,X=X)
    
        K_x = torch.autograd.functional.jacobian(evaluate_model.model_der,inputs=(parameters_eval))
        S_hat = torch.inverse( K_x.T @ ( s_e_inverse @ K_x ) + s_a_inverse  )
        G_hat = torch.inverse( s_a_inverse + K_x.T @ ( s_e_inverse @ K_x  )  ) @ (  K_x.T @ s_e_inverse  )
        A_hat = G_hat@K_x

        
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
        bbp_derivative = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))
        bbp_two_derivative = torch.empty((3,3,3))
        bbp_values = bbp_values
        for j,i in enumerate([1,2,4]):
            evaluate_model.axis = i
            bbp_two_derivative[j] = torch.autograd.functional.hessian(evaluate_model.bbp_der,inputs=(parameters_eval))
        bbp_delta = error_propagation(bbp_derivative,S_hat,bbp_two_derivative)
        bbp_hat[::2] = bbp_values
        bbp_hat[1::2] = torch.sqrt(bbp_delta)
        evaluate_model.axis=None
    
        output = {'X_hat':past_pred[last_i-1].clone().detach(),'S_hat':S_hat.clone().detach(),'K_hat':K_x.clone().detach(),'G_hat':G_hat.clone().detach(),'A_hat':A_hat.clone().detach(),\
                  'Y_hat':pred.clone().detach(),'kd':kd_hat.clone().detach(),'bbp':bbp_hat.clone().detach(),'RRS':last_rrs.clone().detach().numpy()}
    
        print("time for training...",time.time() - time_init)
        
        return ls_val,past_pred[:last_i].clone().detach(),output
    else:
        return last_rrs.clone().detach().numpy(),past_pred[last_i-1].clone().detach().numpy()


class custom_Loss(nn.Module):

    def __init__(self,x_a,s_a,s_e,precision = torch.float32):
        super(custom_Loss, self).__init__()
        self.x_a = x_a
        self.s_a = s_a
        self.s_e = s_e
        self.s_e_inverse = torch.inverse(s_e)
        self.s_a_inverse = torch.inverse(s_a)
        self.precision = precision


    def forward(self,y,f_x,x,test = False):
        if test == True:
            print(torch.dot( (x - self.x_a), self.s_a_inverse @ (x - self.x_a)),torch.dot((y - f_x),self.s_e_inverse @ (y - f_x )))
        return  (   torch.dot((y - f_x),self.s_e_inverse @ (y - f_x )) + torch.dot( (x - self.x_a), self.s_a_inverse @ (x - self.x_a))   ).to(self.precision)

if __name__ == '__main__':

    initial_conditions_path = '/Users/carlos/Documents/surface_data_analisis/npy_data/'
    data_path = '/Users/carlos/Documents/surface_data_analisis/npy_data/'
    train_data = customTensorData(data_path='./npy_data',which='train')
    test_data = customTensorData(data_path='./npy_data',which='test')

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    s_e = torch.tensor(np.load('./bayes_parameters/s_e.npy')).to(torch.float32)
    x_a = torch.tensor(np.load('./bayes_parameters/x_a.npy')).to(torch.float32)
    s_a = torch.tensor(np.load('./bayes_parameters/s_a.npy')).to(torch.float32)
    output_file = '/Users/carlos/Documents/surface_data_analisis/results_bayes'
    #output_file = '.'
    s_e = torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4])#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf 


    #mps_device = torch.device("mps")
    
    model = MODEL().to('cpu')
    loss = custom_Loss(x_a,s_a,s_e)

    evaluation_time = []

    X,Y = train_data.__getitem__(0)
    #print(X,Y)
    X,Y = next(iter(train_dataloader))
    #print(X,Y)

    for i in range(1):

    
        X,Y = train_data.__getitem__(i)
        init_time = time.time()
        
        print(loss(Y,model(X),Y[:3]))

        evaluation_time.append(time.time() - init_time)

    print(np.mean(evaluation_time))




