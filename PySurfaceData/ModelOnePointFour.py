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
import matplotlib
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
from matplotlib import ticker

import warnings



class customTensorData():

    def __init__(self,data_path='./npy_data',transform=None,target_transform=None,train_percentage = 0.9,from_where = 'left',randomice=False,specific_columns=None,which='train',\
                 seed=None,per_day=True,precision=torch.float32,one_dimensional = False):
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
        Parameters:
          which: can be 'train', 'test' or 'all', if 'tran' or 'test', torch.Dataloader will use only a percentage of the total data, which can be randomly ordered or not, depending if
          randomice is equal to False or True. If randomice is False, will use the 'train_percentage' percentage of the data for the training data, starting from 'from_where',
          and the remaining for the test data. If randomice is True, the data is shuffled before the partition is made. Use seed if you whant to have the same partition for train and test data.
          Default 'train'.

          train_percentage: The percentage of the total data used for train data. A number between 0 and 1. Default 0.9.

          from_where: When randomice is Falce, defines if the train data is the first train_percentage percentage of the data ('left') or the last percentage of the data ('right'). Default, 'left'.

          seed: random seed used for the shuffle of the data. Default None

          per_day: if True, then RRS is the output Y of __getitem__, if False, the in-situ data is the output Y. Default True

          precision: the precision for the tensores to use. Default torch.float32

          one_dimensional: If the function __getitem__ will return one or two dimentional tensors. I False, the output will be matrices with 5 rows, one for each wavelenght. Default False

          randomice: if the data is shuffle before returned. Default False.

        Variables:

          one_dimensional: the input of parameter one_dimensional
          dates: the dates as number of days since year 2000, month 1, day 1, for the set of data selected. If randomiced = True, dates is equally randomiced.
          init: can be used as initial conditions for chla, NAP and CDOM, are the values from a first run. Could increase the performace, but also induce a posible bias.
          x_data: the input data for the model.
          x_column_names: if one_dimensional=True, is the name of the columns of x_data. If not, the same data is redestributed in matrices with 5 rows, one for each wavelenght, with zenith
          and PAR being columns with the same value repeated 5 times.
          y_data: the output data for the model.
          y_column_names: Is the name of the columns of y_data.
          per_day: value of the parameter per_day
          init_column_names: name of the initial conditions.
          len_data: lenght of all the data used, is diferent for which='Train', 'Test' or 'all'.
          indexes: a list with numbers from 0 to the lenght of all the data.
          my_indexes: the list 'indexes' reshufled if randomice=True, and cut depending on the value of 'which'.
          test_indexes: the list 'indexes' reshufled if randomice=True, and cut for the test data.
          train_indexes: the list 'indexes' reshufled if randomice=True, and cut for the train data.

        """

        x_data = np.load(data_path + '/x_data_all.npy',allow_pickle=False)
        self.dates = x_data[:,-1]
        self.init = x_data[:,22:25]
        self.x_data = np.delete(x_data,[22,23,24,25],axis=1)

        self.one_dimensional = one_dimensional

        self.per_day = per_day
        if self.per_day == True:
            self.y_data = self.x_data[:,:5]
            self.y_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555']
            self.x_data = np.delete(self.x_data,[0,1,2,3,4],axis=1)
            self.x_column_names = ['Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','lambda_412','lambda_442',\
                               'lambda_490','lambda_510','lambda_555','zenith','PAR']
        else:
            self.y_data = np.load(data_path + '/y_data_all.npy',allow_pickle=False)
            self.y_column_names = ['chla','kd_412','kd_442','kd_490','kd_510','kd_555','bbp_442','bbp_490','bbp_555']
            self.x_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555','Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','lambda_412','lambda_442',\
                               'lambda_490','lambda_510','lambda_555','zenith','PAR']

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
            self.my_indexes = self.test_indexes
        self.precision=precision
        self.len_data = len(self.my_indexes)
        self.dates = self.dates[self.my_indexes]
        self.init = self.init[self.my_indexes]
        self.x_std = np.delete(np.nanstd(self.x_data[self.train_indexes],axis=0),[-3,-4,-5,-6,-7])
        self.y_std = np.nanstd(self.y_data[self.train_indexes],axis=0)
        self.x_mean = np.delete(np.nanmean(self.x_data[self.train_indexes],axis=0),[-3,-4,-5,-6,-7])
        self.y_mean = np.nanmean(self.y_data[self.train_indexes],axis=0)
        self.x_normilized = (np.delete(self.x_data,[-3,-4,-5,-6,-7],axis=1) - self.x_mean)/self.x_std
        self.y_normilized = (self.y_data - self.y_mean)/self.y_std
        
    def __len__(self):

            return len(self.my_indexes)

    def __getitem__(self, idx):
        if self.one_dimensional == False:
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
                image[:,2] = torch.tensor(self.x_data[self.my_indexes][idx][10:15])
                image[:,3] = torch.tensor(self.x_data[self.my_indexes][idx][15:20])
                image[:,4] = torch.tensor(self.x_data[self.my_indexes][idx][20])
                image[:,5] = torch.tensor(self.x_data[self.my_indexes][idx][21])
        else:

            label = torch.tensor(self.y_normilized[self.my_indexes][idx]).unsqueeze(0)
            image = torch.tensor(self.x_normilized[self.my_indexes][idx]).unsqueeze(0)
          
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.to(self.precision), label.to(self.precision)




def read_constants(file1='./cte_lambda.csv',file2='./cst.csv',tensor = True,my_device = 'cpu'):
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
        if tensor == True:
            constant[key] = torch.tensor(list(constant[key].values()),dtype=torch.float32).to(my_device)
        else:
            constant[key] = np.array(list(constant[key].values()))
    cts = pd.read_csv(file2)
        
    for i in range(len(cts['name'])):
        constant[cts['name'].iloc[i]] = cts['value'].iloc[i]
        
    lambdas = np.array([412.5,442.5,490,510,555]).astype(float)
    
    linear_regression=stats.linregress(lambdas,constant['scattering_PH'].cpu())
    linear_regression_slope = linear_regression.slope
    linear_regression_intercept = linear_regression.intercept
    constant['linear_regression_slope_s'] = linear_regression_slope
    constant['linear_regression_intercept_s'] = linear_regression_intercept

    linear_regression=stats.linregress(lambdas,constant['backscattering_PH'].cpu())
    linear_regression_slope = linear_regression.slope
    linear_regression_intercept = linear_regression.intercept
    constant['linear_regression_slope_b'] = linear_regression_slope
    constant['linear_regression_intercept_b'] = linear_regression_intercept
    
    return constant





###################################################################################################################################################################################################
############################################################################FUNCTIONS NEEDED TO DEFINE THE FORWARD MODEL###########################################################################
###################################################################################################################################################################################################

################Functions for the absortion coefitient####################
def absortion_CDOM(lambda_,perturbation_factors,tensor = True,constant = None):
    """
    Function that returns the mass-specific absorption coefficient of CDOM, function dependent of the wavelength lambda. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == False:
        return constant['dCDOM']*np.exp(-(constant['sCDOM'] * perturbation_factors[6])*(lambda_ - 450.))
    else:
        return constant['dCDOM']*torch.exp(-(constant['sCDOM'] * perturbation_factors[6])*(torch.tensor(lambda_) - 450.))

def absortion_NAP(lambda_,tensor = True,constant = None):
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

def absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    Total absortion coeffitient.
    aW,λ (values used from Pope and Fry, 1997), aP H,λ (values averaged and interpolated from
    Alvarez et al., 2022).
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if axis == None:
        if tensor == True:
            return constant['absortion_w'] + (constant['absortion_PH']* perturbation_factors[0])*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor,constant = constant)* perturbation_factors[5])*CDOM + absortion_NAP(lambda_,tensor=tensor,constant = constant)*NAP
        else:
            return constant['absortion_w'] + constant['absortion_PH'] * perturbation_factors[0]*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor,constant = constant)* perturbation_factors[5])*CDOM + absortion_NAP(lambda_,tensor=tensor,constant = constant)*NAP
    else:
        if tensor == True:
            return constant['absortion_w'][axis] + (constant['absortion_PH'][axis] * perturbation_factors[0])*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor,constant = constant)* perturbation_factors[5])*CDOM + absortion_NAP(lambda_,tensor=tensor,constant = constant)*NAP
        else:
            return constant['absortion_w'] + (constant['absortion_PH'][axis] * perturbation_factors[0])*chla + \
                (absortion_CDOM(lambda_, perturbation_factors,tensor=tensor,constant = constant)* perturbation_factors[5])*CDOM + absortion_NAP(lambda_,tensor=tensor,constant = constant)*NAP

##############Functions for the scattering coefitient########################
def Carbon(chla,PAR, perturbation_factors,tensor=True,constant = None):
    """
    defined from the carbon to Chl-a ratio. 
    theta_o, sigma, beta, and theta_min constants (equation and values computed from Cloern et al., 1995), and PAR
    the Photosynthetically available radiation, obtained from the OASIM model, see Lazzari et al., 2021.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    nominator = chla
    beta =  constant['beta'] * perturbation_factors[11]
    sigma = constant['sigma'] * perturbation_factors[12]
    exponent = -(PAR - beta)/sigma
    if tensor == False:
        denominator = (constant['Theta_o']* perturbation_factors[10]) * ( np.exp(exponent)/(1+np.exp(exponent)) ) + \
        (constant['Theta_min'] * perturbation_factors[9])
    else:
        denominator = (constant['Theta_o']* perturbation_factors[10]) * ( torch.exp(exponent)/(1+torch.exp(exponent)) ) + \
        (constant['Theta_min'] * perturbation_factors[9])
    return nominator/denominator

def scattering_ph(lambda_,perturbation_factors,tensor = True,constant = None):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    
    return (constant['linear_regression_slope_s'] * perturbation_factors[1]) *\
        lambda_ + constant['linear_regression_intercept_s'] * perturbation_factors[2]

def backscattering_ph(lambda_,perturbation_factors,tensor = True,constant = None):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    
    return (constant['linear_regression_slope_b'] * perturbation_factors[3]) *\
        lambda_ + constant['linear_regression_intercept_b'] * perturbation_factors[4]

def scattering_NAP(lambda_,tensor=True,constant = None):
    """
    NAP mass-specific scattering coefficient.
    eNAP and fNAP constants (equation and values used from Gallegos et al., 2011)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return constant['eNAP']*(550./lambda_)**constant['fNAP']

def scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    Total scattering coefficient.
    bW,λ (values interpolated from Smith and Baker, 1981,), bP H,λ (values used
    from Dutkiewicz et al., 2015)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if axis == None:
        if tensor == True:
            return constant['scattering_w'] + scattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant) + \
                scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
        else:
            return constant['scattering_w'] + scattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant) + \
                scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
    else:
        if tensor == True:
            return constant['scattering_w'][axis] + (scattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[axis] + \
                scattering_NAP(lambda_,tensor=tensor,constant = constant)[axis] * NAP
        else:
            return constant['scattering_w'][axis] + (scattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[axis] + \
                scattering_NAP(lambda_,tensor=tensor,constant = constant)[axis] * NAP

#################Functions for the backscattering coefitient#############

def backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if axis == None:
        if tensor == True:
            return constant['backscattering_w'] + backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant) + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
        else:
            return constant['backscattering_w'] + backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant) + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
    else:
        if tensor == True:
            return constant['backscattering_w'][axis] + (backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[axis] + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
        else:
            return constant['backscattering_w'][axis] + (backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[axis] + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP



###############Functions for the end solution of the equations###########
#The final result is written in terms of these functions, see ...

def c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,my_device = 'cpu',constant = None): 
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
        return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant))/torch.cos(torch.tensor(zenith)*3.1416/180)
    else:
    	return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant))/np.cos(zenith*3.1416/180)

def F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant))/\
        torch.cos(torch.tensor(zenith)*3.1416/180.)
    else:
    	return (scattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant))/\
        np.cos(zenith*3.1416/180.)

def B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor == True:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant)/torch.cos(torch.tensor(zenith)*3.1416/180) 
    else:
    	return  constant['rd']*backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant)/np.cos(zenith*3.1416/180)

def C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) )/\
        constant['vs']

def B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant))/constant['vu']

def B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (constant['rs'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant))/constant['vs']

def C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (absortion(lambda_,chla,CDOM,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + constant['ru'] * backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant))/\
        constant['vu']

def D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return (0.5) * (C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + \
                    ((C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant))**2 -\
                     4. * B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) )**(0.5))

def x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant)
    nominator = -(C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)) *\
        F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) -\
        B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant)

    return nominator/denominator

def y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) - C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)) +\
        B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * B_u(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant)
    nominator = (-B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * F_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant) ) +\
        (-C_s(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)) *\
        B_d(lambda_,zenith,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant)

    return nominator/denominator

def C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return E_dif_o - x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * E_dir_o

def r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return B_s(lambda_,PAR,chla,NAP,perturbation_factors,tensor=tensor,axis=axis,constant = constant)/D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)

def k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis = None,constant = None):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return D(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) - C_u(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)


def E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
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
        return E_dir_o*np.exp(-z*c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant))
    else:
        return E_dir_o*torch.exp(-z*c_d(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant))

def E_u(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True,axis=None,constant = None):
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

        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)*\
                np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)*z)+\
                y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant) * r_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant)*\
                torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant)*z)+\
                y(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant)

def E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis = None,constant = None):
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

        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) *\
                np.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)*z)+\
                x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis=axis,constant = constant)
    else:
        return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant) *\
                torch.exp(- k_plus(lambda_,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant)*z)+\
                x(lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant) * E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant)
        

def bbp(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
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
            return backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant) + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
        else:
            return backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant) + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
    else:
        if tensor == True:
            return (backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[axis] + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP
        else:
            return (backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[axis] + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP

def kd(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=True,axis=None,constant = None):
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
        return (z**-1)*np.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis = axis,constant = constant) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis = axis,constant = constant)))
    else:
        return (z**-1)*torch.log((E_dir_o + E_dif_o)/(E_dir(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant) +\
                                                  E_dif(z,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,axis=axis,constant = constant)))

##########################from the bio-optical model to RRS(Remote Sensing Reflectance)##############################
#defining Rrs
#Q=5.33*np.exp(-0.45*np.sin(np.pi/180.*(90.0-Zenith)))

def Q_rs(zenith,perturbation_factors,tensor=True,constant = None):
    """
    Empirical result for the Radiance distribution function, 
    equation from Aas and Højerslev, 1999, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    if tensor==True:
        return (5.33 * perturbation_factors[7])*torch.exp(-(0.45 * perturbation_factors[8])*torch.sin((3.1416/180.0)*(90.0-torch.tensor(zenith))))
    else:
        return  (5.33 * perturbation_factors[7])*np.exp(-(0.45 * perturbation_factors[8])*np.sin((3.1416/180.0)*(90.0-zenith)))

def Rrs_minus(Rrs,tensor=True,constant = None):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return Rrs/(constant['T']+constant['gammaQ']*Rrs)

def Rrs_plus(Rrs,tensor=True,constant = None):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return Rrs*constant['T']/(1-constant['gammaQ']*Rrs)

def Rrs_MODEL(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor = True,axis = None,constant = None):
    """
    Remote Sensing Reflectance.
    Aas and Højerslev, 1999.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    Rrs = E_u(0,E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM,perturbation_factors,tensor=tensor,axis = axis,constant = constant)  /  (   Q_rs(zenith,perturbation_factors,tensor=tensor,constant = constant)*(E_dir_o + E_dif_o)   )
    return Rrs_plus( Rrs ,tensor = tensor,constant = constant)


###################################################################################################################################################################################################
#########################################################################FUNCTIONS NEEDED TO FIND THE OPTICAL CONSTITUENTS#########################################################################
###################################################################################################################################################################################################
                
class Forward_Model(nn.Module):
    """
    Bio-Optical model plus corrections, in order to have the Remote Sensing Reflectance, in terms of the inversion problem. 
    Forward_Model(x) returns a tensor, with each component being the Remote Sensing Reflectance for each given wavelength. 
    if the data has 5 rows, each with a different wavelength, RRS will return a vector with 5 components.  RRS has tree parameters, 
    self.chla is the chlorophil-a, self.NAP the Non Algal Particles, and self.CDOM the Colored Dissolved Organic Mather. 
    According to the invention problem, we want to estimate them by making these three parameters have two constraints,
    follow the equations of the bio-optical model, plus, making the RRS as close as possible to the value
    measured by the satellite.
    
    """
    def __init__(self,precision = torch.float32,num_days=1,learning_chla = True,learning_perturbation_factors = False):
        super().__init__()
        if learning_chla == True:
            self.chparam = nn.Parameter(torch.rand((num_days,1,3), dtype=torch.float32), requires_grad=True)
        self.learning_chla = learning_chla

        if learning_perturbation_factors == False:
            self.perturbation_factors = torch.ones(14, dtype=torch.float32)
        else:
            self.perturbation_factors =  nn.Parameter(torch.ones(14, dtype=torch.float32), requires_grad=True)

        self.perturbation_factors_names = [
            '$\epsilon_{a,ph}$',
            '$\epsilon_{tangent,s,ph}$',
            '$\epsilon_{intercept,s,ph}$',
            '$\epsilon_{tangent,b,ph}$',
            '$\epsilon_{intercept,b,ph}$',
            '$\epsilon_{a,cdom}$',
            '$\epsilon_{exp,cdom}$',
            '$\epsilon_{q,1}$',
            '$\epsilon_{q,2}$',
            '$\epsilon_{theta,min}$',
            '$\epsilon_{theta,o}$',
            '$\epsilon_\\beta$',
            '$\epsilon_\sigma$',
            '$\epsilon_{b,nap}$',
        ]
        self.precision = precision

    def forward(self,x_data,parameters = None, axis = None,perturbation_factors_ = None,constant = None):
        """
        x_data: pandas dataframe with columns [E_dif,E_dir,lambda,zenith,PAR].
        """
        if type(perturbation_factors_) == type(None):
            perturbations = self.perturbation_factors
        else:
            perturbations = perturbation_factors_
        if type(parameters) == type(None):
            if self.learning_chla == False:
                print('Please provide a tensor with the value of chla,nap and cdom')

            if type(axis) == type(None):
            
                Rrs = Rrs_MODEL(x_data[:,:,0],x_data[:,:,1],x_data[:,:,2],\
                                x_data[:,:,3],x_data[:,:,4],self.chparam[:,:,0],self.chparam[:,:,1],self.chparam[:,:,2],perturbations,constant = constant)
            
                return Rrs.to(self.precision)
            else:

                Rrs = Rrs_MODEL(x_data[:,axis,0],x_data[:,axis,1],x_data[:,axis,2],\
                                x_data[:,axis,3],x_data[:,axis,4],self.chparam[:,:,0],self.chparam[:,:,1],self.chparam[:,:,2],perturbations,constant = constant)
            
                return Rrs.to(self.precision)

        else:
            if type(axis) == type(None):
                
                Rrs = Rrs_MODEL(x_data[:,:,0],x_data[:,:,1],x_data[:,:,2],\
                                x_data[:,:,3],x_data[:,:,4],parameters[:,:,0],parameters[:,:,1],parameters[:,:,2],perturbations,constant = constant)
            
                return Rrs.to(self.precision)
            else:
                Rrs = Rrs_MODEL(x_data[:,axis,0],x_data[:,axis,1],x_data[:,axis,2],\
                                x_data[:,axis,3],x_data[:,axis,4],parameters[:,:,0],parameters[:,:,1],parameters[:,:,2],perturbations,constant = constant)
                return Rrs.to(self.precision)
                
def error_propagation(df,sigma):

    error_ = df @  sigma @ torch.transpose(df,1,2)
    error = torch.diagonal(error_,dim1=1,dim2=2)
    return error

class evaluate_model_class():
    """
    class to evaluate functions needed to compute the uncerteinty. 
    """
    def __init__(self,model,X,axis=None,constant = None):
        self.axis = axis
        self.model = model
        self.X = X
        self.constant = constant
    def model_der(self,parameters,perturbation_factors_ = None):
        
        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        return self.model(self.X,parameters = parameters,axis = self.axis,perturbation_factors_ = perturbations,constant = self.constant)
        
    def kd_der(self,parameters,perturbation_factors_ = None):
        
        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        if self.axis == None:
            kd_values = kd(9,self.X[:,:,0],self.X[:,:,1],self.X[:,:,2],\
                           self.X[:,:,3],self.X[:,:,4],parameters[:,:,0],parameters[:,:,1],parameters[:,:,2],perturbations,constant = self.constant)
            return kd_values
        else:
            kd_values = kd(9,self.X[:,self.axis,0],self.X[:,self.axis,1],self.X[:,self.axis,2],\
                           self.X[:,self.axis,3],self.X[:,self.axis,4],parameters[:,:,0],parameters[:,:,1],parameters[:,:,2],perturbations,axis = self.axis,constant = self.constant)
            return kd_values

    def bbp_der(self,parameters,perturbation_factors_ = None):

        if type(perturbation_factors_) == type(None):
            perturbations = self.model.perturbation_factors
        else:
            perturbations = perturbation_factors_
            
        if self.axis == None:
            bbp_values = bbp(self.X[:,:,0],self.X[:,:,1],self.X[:,:,2],\
                             self.X[:,:,3],self.X[:,:,4],parameters[:,:,0],parameters[:,:,1],parameters[:,:,2],perturbations,constant = self.constant)
            return bbp_values[:,[1,2,4]]
        else:
            bbp_values = bbp(self.X[:,self.axis,0],self.X[:,self.axis,1],self.X[:,self.axis,2],\
                             self.X[:,self.axis,3],self.X[:,self.axis,4],parameters[:,:,0],parameters[:,:,1],parameters[:,:,2],perturbations,axis=self.axis,constant = self.constant)
            return bbp_values
            
def train_loop(data_i,model,loss_fn,optimizer,N,kind='all',num_days=1,my_device = 'cpu',constant = None,perturbation_factors_ = None, scheduler = True):
    """
    The train loop evaluates the Remote Sensing Reflectance RRS for each wavelength>>>pred=model(data_i), evaluates the loss function
    >>>loss=loss_fn(pred,y), force the value of the parameters (chla,NAP,CDOM) to be positive, evaluates the gradient of RRS with respect
    to the parameters, >>>loss.backward(), modifies the value of the parameters according to the optimizer criterium, >>>optimizer.step(),
    sets the gradient of RRS to cero, and prints the loss for a given number of iterations. This procedure is performed N times or untyl a treshold is ashieved. 
    After N iterations, it returns two lists with the evolution of the loss function and the last evaluation of the model. 
    """

    ls_val=[]
    past_pred=torch.empty((N,num_days,3))

    time_init = time.time()

    criterium = 1
    criterium_2 = 0
    i=0
    if scheduler == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    X = data_i[0].to(my_device)
    Y = data_i[1].to(my_device)
    s_a = (loss_fn.s_a)
    s_e = (loss_fn.s_e)
    s_e_inverse = (loss_fn.s_e_inverse)
    s_a_inverse = (loss_fn.s_a_inverse)

    dR = (1e-13)

    while (((criterium >dR ) & (i<N)) or ((criterium_2 < 100)&(i<N))):
        
        pred = model(X,constant = constant,perturbation_factors_ = perturbation_factors_)
        loss = loss_fn(Y,pred,model.state_dict()['chparam'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        for index_,p in enumerate(model.parameters()):
                p.data.clamp_(0)
                
        ls_val.append(loss.item())
        past_pred[i] = model.state_dict()['chparam'][:,0,:]
        
        if i != 0:
            criterium = ls_val[-2] - ls_val[-1]
        if criterium <=dR:
            criterium_2+=1
        i+=1
        if scheduler == True:
            scheduler.step(loss)
    last_i = i
    last_rrs = pred.clone().detach()
    if kind == 'all':
        parameters_eval = list(model.parameters())[0].clone().detach()
        evaluate_model = evaluate_model_class(model=model,X=X,constant = constant)
        
        K_x = torch.empty((len(parameters_eval),5,3))
        K_x_ = torch.autograd.functional.jacobian(evaluate_model.model_der,inputs=(parameters_eval))
        for i in range(len(parameters_eval)):
            K_x[i] = torch.reshape(K_x_[i,:,i,:,:],(5,3))
            
        S_hat = torch.inverse( torch.transpose(K_x,1,2) @ ( s_e_inverse @ K_x ) + s_a_inverse  )

        X_hat = np.empty((len(parameters_eval),6))
        X_hat[:,::2] = past_pred[last_i-1].clone().detach()
        X_hat[:,1::2] = torch.sqrt(torch.diagonal(S_hat,dim1=1,dim2=2).clone().detach())
       
        kd_hat = torch.empty((len(parameters_eval),10))
        bbp_hat = torch.empty((len(parameters_eval),6))
        bbp_index = 0

        kd_values = evaluate_model.kd_der(parameters_eval)
        kd_derivative = torch.empty((len(parameters_eval),5,3))
        kd_derivative_ = torch.autograd.functional.jacobian(evaluate_model.kd_der,inputs=(parameters_eval))
        for i in range(len(parameters_eval)):
            kd_derivative[i] = torch.reshape(kd_derivative_[i,:,i,:,:],(5,3))
               
        kd_delta = error_propagation(kd_derivative,S_hat)
        kd_hat[:,::2] = kd_values.clone().detach()
        kd_hat[:,1::2] = torch.sqrt(kd_delta).clone().detach()
    
        bbp_values = evaluate_model.bbp_der(parameters_eval)
        bbp_derivative = torch.empty((len(parameters_eval),3,3))
        bbp_derivative_ = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))
        for i in range(len(parameters_eval)):
            bbp_derivative[i] = torch.reshape(bbp_derivative_[i,:,i,:,:],(3,3))
        
        bbp_delta = error_propagation(bbp_derivative,S_hat)
        bbp_hat[:,::2] = bbp_values.clone().detach()
        bbp_hat[:,1::2] = torch.sqrt(bbp_delta).clone().detach()
    
        output = {'X_hat':X_hat,'kd_hat':kd_hat,'bbp_hat':bbp_hat,'RRS_hat':last_rrs}
    
        print("time for training...",time.time() - time_init)
        return output
    elif kind == 'parameter_estimation':
        
        evaluate_model = evaluate_model_class(model=model,X=X,constant = constant)
        
        parameters_eval = list(model.parameters())[0]
        
        output = torch.empty((X.shape[0],9))
        output[:,0] = past_pred[last_i-1][:,0]
        output[:,1:6] = evaluate_model.kd_der(parameters_eval,perturbation_factors_ = perturbation_factors_)
        output[:,6:] = evaluate_model.bbp_der(parameters_eval,perturbation_factors_ = perturbation_factors_)

        return output

    else:
        print("time for training...",time.time() - time_init)
        return last_rrs.clone().detach().numpy(),past_pred[last_i-1].clone().detach().numpy()


def initial_conditions(data_,batch_size,F_model):

    initial_conditions = torch.tensor(data_.init[:batch_size]).unsqueeze(1)
    initial_conditions[initial_conditions == 0] == torch.rand(1)
    
    state_dict = F_model.state_dict()
    state_dict['chparam'] = torch.ones((batch_size,1,3), dtype=torch.float32)*initial_conditions
    F_model.load_state_dict(state_dict)


    
    
class custom_Loss(nn.Module):

    def __init__(self,x_a,s_a,s_e,precision = torch.float32,num_days=1,my_device = 'cpu'):
        super(custom_Loss, self).__init__()
        self.x_a = torch.stack([x_a for _ in range(num_days)]).to(my_device)
        self.s_a = s_a.to(my_device)
        self.s_e = s_e.to(my_device)
        self.s_e_inverse = torch.inverse(self.s_e)
        self.s_a_inverse = torch.inverse(self.s_a)
        self.precision = precision


    def forward(self,y,f_x,x,test = False):
        if test == True:
            print(  (y - f_x) @ ( self.s_e_inverse @ (y - f_x ).T ),(x[:,0,:] - self.x_a) @( self.s_a_inverse @ (x[:,0,:] - self.x_a).T ))
        return  torch.trace(   (y - f_x) @ ( self.s_e_inverse @ (y - f_x ).T ) +  (x[:,0,:] - self.x_a) @( self.s_a_inverse @ (x[:,0,:] - self.x_a).T )  ).to(self.precision)


class Parameter_Estimator(nn.Module):
    """
	Model that attempts to learn the perturbation factors. 
    """
    def __init__(self):
        super().__init__()
        self.perturbation_factors = nn.Parameter(torch.ones(14, dtype=torch.float32), requires_grad=True)

        self.perturbation_factors_names = [
            '$\epsilon_{a,ph}$',
            '$\epsilon_{tangent,s,ph}$',
            '$\epsilon_{intercept,s,ph}$',
            '$\epsilon_{tangent,b,ph}$',
            '$\epsilon_{intercept,b,ph}$',
            '$\epsilon_{a,cdom}$',
            '$\epsilon_{exp,cdom}$',
            '$\epsilon_{q,1}$',
            '$\epsilon_{q,2}$',
            '$\epsilon_{theta,min}$',
            '$\epsilon_{theta,o}$',
            '$\epsilon_\\beta$',
            '$\epsilon_\sigma$',
            '$\epsilon_{b,nap}$',
    ]

    def forward(self,data,constant,model,loss,optimizer,num_iterations=100,batch_size=1,my_device='cpu',scheduler = False):

        return train_loop(data,model,loss,optimizer,num_iterations,kind='parameter_estimation',num_days = batch_size,\
                          my_device = my_device,constant = constant,perturbation_factors_ = self.perturbation_factors, scheduler = scheduler)


class custom_Loss_Parameters(nn.Module):

    def __init__(self,precision = torch.float32,my_device = 'cpu'):
        super(custom_Loss_Parameters, self).__init__()
        self.precision = precision


    def forward(self,Y_l,pred_l,nan_array):
        custom_array = ((Y_l-pred_l))**2
        lens = torch.tensor([len(element[~element.isnan()]) for element in nan_array])

        means_output = custom_array.sum(axis=1)/lens
        return means_output.mean().to(self.precision)

def test_model(test_data_,test_X,test_Y,test_Y_nan,x_a,s_a,s_e,constant,test_num_days,perturbation_factors_,parameter_loss,my_device='cpu',test_lr=1e-3):
    test_model = Forward_Model(num_days = test_num_days).to(my_device)
    initial_conditions(test_data_,test_num_days,test_model)
    test_loss = custom_Loss(x_a,s_a,s_e,num_days=test_num_days,my_device = my_device)
    test_optimizer = torch.optim.Adam(test_model.parameters(),lr=test_lr)
    
    test_output = train_loop(test_X,test_model,test_loss,test_optimizer,2000,kind='parameter_estimation',\
                             num_days=test_num_days,constant = constant,perturbation_factors_ = perturbation_factors_, scheduler = True)

    test_output = torch.masked_fill(test_output,torch.isnan(test_Y_nan),0)
    for param in test_model.parameters():
        param.requires_grad = False
            
    test_loss_value = parameter_loss(test_Y,test_output,test_Y_nan)
    return test_loss_value
    
def track_parameters(data_path = '/Users/carlos/Documents/surface_data_analisis/npy_data/',output_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data',iterations=101 ):
    global_init_time = time.time()
    
    data = customTensorData(data_path=data_path,which='train',per_day = False,randomice=True,seed=1853)
    
    test_data = customTensorData(data_path='./npy_data',which='test',per_day = False,randomice=True,seed=1853)
    test_dataloader = DataLoader(test_data, batch_size=test_data.len_data, shuffle=False)

    #mps_device = torch.device("mps")
    my_device = 'cpu'

    
    constant = read_constants(file1='cte_lambda.csv',file2='cst.csv',my_device = my_device)
    
    x_a = torch.ones(3)
    s_a = torch.eye(3)*23
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

    lr = 1e-3
    batch_size = data.len_data
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    model = Forward_Model(num_days=batch_size).to(my_device)
    initial_conditions(data,batch_size,model) #carefull with this step
    loss = custom_Loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)



    Parameter_lr = 0.005
    Parameter_model = Parameter_Estimator()
    
    (p_X,p_Y_nan) = next(iter(dataloader))
    p_X = (p_X[:,:,1:],p_X[:,:,0])
    p_Y = torch.masked_fill(p_Y_nan,torch.isnan(p_Y_nan),0)

    (test_X,test_Y_nan) = next(iter(test_dataloader))
    test_X = (test_X[:,:,1:],test_X[:,:,0])
    test_Y = torch.masked_fill(test_Y_nan,torch.isnan(test_Y_nan),0)
    
    
    Parameter_loss = custom_Loss_Parameters()
    Parameter_optimizer = torch.optim.Adam(Parameter_model.parameters(),lr=Parameter_lr)
    
    p_ls = []

    p_past_parameters = torch.empty((iterations,14))

    scheduler_parameters = torch.optim.lr_scheduler.ReduceLROnPlateau(Parameter_optimizer, 'min')
    test_loss = []
    for i in range(iterations):

        parameters_iter_time = time.time()
        
        for param in model.parameters():
            param.requires_grad = True
        p_pred = Parameter_model(p_X,constant,model,loss,optimizer,batch_size = batch_size,num_iterations=10, scheduler = False)
        p_pred = torch.masked_fill(p_pred,torch.isnan(p_Y_nan),0)
        for param in model.parameters():
            param.requires_grad = False
            
        
        p_loss = Parameter_loss(p_Y,p_pred,p_Y_nan)
        p_loss.backward()
        for param in Parameter_model.parameters():#seting nan gradients to cero (only move in direction where I have information)
            p_grad = param.grad
            p_grad[p_grad != p_grad ] = 0

        Parameter_optimizer.step()
        Parameter_optimizer.zero_grad()
        
        for index_,p in enumerate(Parameter_model.parameters()):
            p.data.clamp_(min=0.1,max=1.9)
                
        p_ls.append(p_loss.item())
        p_past_parameters[i] =  next(iter(Parameter_model.parameters()))
        scheduler_parameters.step(p_loss)
        if i % 10 == 0:
            test_loss_i = test_model(test_data,test_X,test_Y,test_Y_nan,x_a,s_a,s_e,constant,test_data.len_data,\
                                     Parameter_model.perturbation_factors.clone().detach(),Parameter_loss,my_device='cpu',test_lr=1e-3)
            test_loss.append(test_loss_i.clone().detach().numpy())
            print(i,'loss: ',test_loss_i)
        
    print('Total time: ',time.time() - global_init_time )
    to_plot = p_past_parameters.clone().detach().numpy()

    np.save(output_path + '/past_parameters.npy',to_plot )
    np.save(output_path + '/test_loss_each_10.npy',np.array(test_loss) )
    np.save(output_path + '/train_loss.npy',np.array(p_ls) )

def plot_tracked_parameters(data_path='/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data', perturbation_factors_history = [], test_loss = [], train_loss = [],\
                            save = True,save_file_name = '/perturbation_factors.pdf',color_palet = 1, with_loss = False, side='right'):

    if len(perturbation_factors_history) > 0 :
        to_plot = perturbation_factors_history
    else:
        to_plot = np.load(data_path + '/past_parameters.npy')[:1500]
    if len(test_loss)>0:
        test_loss = test_loss
    else:
        test_loss = np.load(data_path + '/test_loss_each_10.npy')[:int(1500/10)]
    if len(train_loss) > 0:
        p_ls = train_loss
    else:
        p_ls = np.load(data_path + '/train_loss.npy')[:1500]
    iterations = len(to_plot)
    constant = read_constants(file1='./cte_lambda.csv',file2='./cst.csv')
    
    perturbation_factors_names = [
            '$\epsilon_{a,ph}$',
            '$\epsilon_{tangent,s,ph}$',
            '$\epsilon_{intercept,s,ph}$',
            '$\epsilon_{tangent,b,ph}$',
            '$\epsilon_{intercept,b,ph}$',
            '$\epsilon_{a,cdom}$',
            '$\epsilon_{exp,cdom}$',
            '$\epsilon_{q,1}$',
            '$\epsilon_{q,2}$',
            '$\epsilon_{theta,min}$',
            '$\epsilon_{theta,o}$',
            '$\epsilon_\\beta$',
            '$\epsilon_\sigma$',
            '$\epsilon_{b,nap}$',
    ]
    if color_palet == 1:
        colors = plt.cm.ocean(np.linspace(0,1,17))
        cmap_ = plt.cm.Blues
    elif color_palet == 2:
        colors = plt.cm.copper(np.linspace(-1,1,14))
        cmap_ = plt.cm.Oranges
        
    linestyle_tuple = [
        ('lt1',        (0, (1, 6))),
        ('loosely dotted',        (0, (6,1,1,1))),
        ('asdf',                (0, (1, 2))),
        ('densely dashdotdotted', (0, (3,2,3,2,1,2))),
        ('densely dotted',        (0, (1, 0.5))),
        ('long dash with offset', (0, (3, 1))),
        ('loosely dashed',        (0, (3, 1,1,1))),
        ('dashed',                (0, (3, 1,1,1,1,1))),
        ('densely dashed',        (0, (3, 3))),
        
        ('loosely dashdotted',    (0, (6, 2))),
        ('dashdotted',            (0, (6,2,1,2))),
        ('densely dashdotted',    (0, (6,2,3,2))),

        ('dashdotdotted',         (0, (6,1,1,1,1))),
        ('loosely dashdotdotted', (0, (6,1,3,1,3,1))),


    ]

    fig, axs = plt.subplots(ncols = 3, nrows = 2, figsize=(15,10),width_ratios = [1/3,1/3,1/3])
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, :])

    
        
    for i in range(14):
        axbig.plot(np.arange(iterations),to_plot[:,i],label=perturbation_factors_names[i],color = colors[i],linestyle=linestyle_tuple[i][1], alpha = 0.8)
    axbig.set_ylabel('Perturbation factors $\epsilon$',fontsize=20)
    if with_loss == True:
        axbig2 = axbig.twinx()
        axbig2.set_ylabel('Loss Error',fontsize=20)
        spacing = int(len(p_ls)/len(test_loss))
        axbig2.plot(np.arange(0,iterations,spacing),test_loss,color = 'gray',label='Test Error', alpha=0.4, lw=3)
        axbig2.plot(np.arange(0,iterations,1),p_ls,color = 'black',label='Train Error', alpha=0.6, lw=3)
        axbig2.tick_params(axis='y', labelsize=20)
        if side == 'right':
            axbig2.legend(fontsize="14.5",bbox_to_anchor=(1.06,1.),loc='lower right')
        elif side =='left':
            axbig2.legend(fontsize="14.5",bbox_to_anchor=(0.08,1.13),loc='lower right')
        axbig2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
        legend_fontsize_perturbation_factors = "14"
        legend_num_cols = 6

    else:
        legend_fontsize_perturbation_factors = "16"
        legend_num_cols = 7
        
    axbig.set_xlabel('Number of iterations',fontsize=20)
    if side == 'left':
        loc = 'lower right'
        bbox_to_anchor = (1.,1.)

    elif side == 'right':
        loc = 'lower left'
        bbox_to_anchor = (0.,1.)
        if with_loss == False:
            axbig.yaxis.set_label_position("right")
            axbig.yaxis.tick_right()
        
    axbig.legend(fontsize="14.5",bbox_to_anchor=bbox_to_anchor,ncol=legend_num_cols,loc=loc)
    axbig.tick_params(axis='y', labelsize=20)
    axbig.tick_params(axis='x', labelsize=20)
    axbig.set_ylim(0.,1.9)
    


    plot_track_absortion_ph(axs[1,0],constant,to_plot,iterations,cmap=cmap_)
    plot_track_scattering_ph(axs[1,1],constant,to_plot,iterations,cmap=cmap_)
    plot_track_backscattering_ph(axs[1,2],constant,to_plot,iterations,cmap=cmap_)

    if side == 'right':
        cax = plt.axes((0.93,0.075,0.01,0.325))
        loc = 'right'
        if with_loss == True:
            fig.subplots_adjust(right=2)
        else:
            fig.subplots_adjust(right=0.88)

    elif side == 'left':

        loc = 'left'
        if with_loss == True:
            fig.subplots_adjust(left=0.3)
            cax = plt.axes((0.05,0.075,0.01,0.325))
        else:
            fig.subplots_adjust(left=0.18)
            cax = plt.axes((0.07,0.075,0.01,0.325))
    cl = fig.colorbar(cax = cax,mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=iterations), cmap=cmap_),orientation='vertical', fraction=1.,pad=0, location = loc)
    cl.ax.tick_params(labelsize=15)
    cl.set_label(label='Number of iterations',size=15)
    
    if side == 'left':
        positions = [(-0.05,1.1),(-0.1,1.1)]
        texts = ['(a)','(b)','(c)','(d)']
    elif side == 'right':
        positions = [(-0.05,1.1),(-0.1,1.1)]#[(1.05,1.1),(1.1,1.1)]
        texts = ['(e)','(f)','(g)','(h)']
        for ax in axs[1,:]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
    axbig.text(positions[0][0],positions[0][1],texts[0],transform = axbig.transAxes,fontsize='20')
    axs[1,0].text(positions[1][0],positions[1][1],texts[1],transform = axs[1,0].transAxes,fontsize='20')
    axs[1,1].text(positions[1][0],positions[1][1],texts[2],transform = axs[1,1].transAxes,fontsize='20')
    axs[1,2].text(positions[1][0],positions[1][1],texts[3],transform = axs[1,2].transAxes,fontsize='20')

    
    fig.tight_layout()

    #plt.show()
    if save == True:
        plt.savefig(data_path + save_file_name)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_track_absortion_ph(ax_,constant,past_perturbation_factors,len_past,cmap = plt.cm.Blues):

    lambdas = np.array([412.5,442.5,490,510,555])
    storical_absortion_ph =   (   past_perturbation_factors[:,0].reshape((len_past,1))*constant['absortion_PH'].numpy() ).T


    colors = cmap(np.linspace(0,1,15))
            
    for i in range(15):
        ax_.plot(lambdas,storical_absortion_ph[:,int((len_past/15)*i)],color = colors[i])
    ax_.set_xticks(lambdas,['412.5','442.5','490','510','555'])
    ax_.set_xlabel('Wavelenght (nm)',fontsize=15)
    ax_.set_ylabel('$a_{PH}$ $(m^2mgChl^{-1})$',fontsize=15)
    ax_.tick_params(axis='y', labelsize=15)
    ax_.tick_params(axis='x', labelsize=15)
    ax_.set_ylim(0,0.06)
    


def plot_track_scattering_ph(ax_,constant,past_perturbation_factors,len_past, cmap = plt.cm.Blues):

    lambdas = np.array([412.5,442.5,490,510,555])
    storical_scattering_ph =   (   past_perturbation_factors[:,1].reshape((len_past,1))*constant['linear_regression_slope_s']*lambdas\
                                   +   past_perturbation_factors[:,2].reshape((len_past,1))*constant['linear_regression_intercept_s'] ).T

    colors = cmap(np.linspace(0,1,15))
        
    for i in range(15):
        ax_.plot(lambdas,storical_scattering_ph[:,int((len_past/15)*i)],color = colors[i])
    ax_.set_xticks(lambdas,['412.5','445.5','490','510','555'])
    ax_.set_xlabel('Wavelenght (nm)',fontsize=15)
    ax_.set_ylabel('$b_{PH}$ $(m^2mgChl^{-1})$',fontsize=15)
    ax_.tick_params(axis='y', labelsize=15)
    ax_.tick_params(axis='x', labelsize=15)
    ax_.set_ylim(0.015,0.047)
    
def plot_track_backscattering_ph(ax_,constant,past_perturbation_factors,len_past,cmap = plt.cm.Blues):

    lambdas = np.array([412.5,442.5,490,510,555])
    storical_scattering_ph =   (   past_perturbation_factors[:,3].reshape((len_past,1))*constant['linear_regression_slope_b']*lambdas\
                                   +   past_perturbation_factors[:,4].reshape((len_past,1))*constant['linear_regression_intercept_b'] ).T

    colors = cmap(np.linspace(0,1,15))

    for i in range(15):
        ax_.plot(lambdas,storical_scattering_ph[:,int((len_past/15)*i)],color = colors[i])
    ax_.set_xticks(lambdas,['412.5','445.5','490','510','555'])
    ax_.set_xlabel('Wavelenght (nm)',fontsize=15)
    ax_.set_ylabel('$b_{b,PH}$ $(m^2mgChl^{-1})$',fontsize=15)
    ax_.tick_params(axis='y', labelsize=15)
    ax_.tick_params(axis='x', labelsize=15)
    ax_.set_ylim(0.5e-5,5.5e-5)

def track_alphas():
    perturbation_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data'
    data_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/npy_data'

    data = customTensorData(data_path=data_path,which='all',per_day = True,randomice=False)
    perturbation_factors = torch.tensor(np.load(perturbation_path + '/past_parameters.npy')[-1]).to(torch.float32)
    my_device = 'cpu'
    constant = read_constants(file1='cte_lambda.csv',file2='cst.csv',my_device = my_device)

    lr = 1e-3
    x_a = torch.ones(3)
    s_a_ = torch.eye(3)
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
    batch_size = data.len_data
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    for alpha in np.linspace(1,51,20):
        s_a = s_a_*alpha
        model = Forward_Model(num_days=batch_size).to(my_device)
        initial_conditions(data,batch_size,model) #carefull with this step
        loss = custom_Loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
        output = train_loop(next(iter(dataloader)),model,loss,optimizer,4000,kind='all',\
                             num_days=batch_size,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = True)

        output_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/results_bayes_optimized/alphas'
        np.save(output_path + '/X_hat_'+str(alpha)+'.npy',output['X_hat'])
        np.save(output_path + '/kd_hat_'+str(alpha)+'.npy',output['kd_hat'])
        np.save(output_path + '/bbp_hat_'+str(alpha)+'.npy',output['bbp_hat'])
        np.save(output_path + '/RRS_hat_'+str(alpha)+'.npy',output['RRS_hat'])
    
if __name__ == '__main__':

    perturbation_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data'
    data_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/npy_data'
    track_parameters(iterations=1500)
    plot_tracked_parameters(data_path='/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data',save=True,color_palet=2,side = 'left',with_loss =False)
    #plot_track_absortion_ph(data_path = data_path)
    #plot_track_scattering_ph(data_path = data_path)
    #plot_track_backscattering_ph(data_path = data_path)


    #test_loss = np.load(data_path + '/test_loss_each_10.npy')[:int(1500/10)]
    #p_ls = np.load(data_path + '/train_loss.npy')[:1500]
    #iterations = len(to_plot)
    #constant = read_constants(file1='./cte_lambda.csv',file2='./cst.csv')
    data = customTensorData(data_path=data_path,which='all',per_day = True,randomice=False)
    perturbation_factors = torch.tensor(np.load(perturbation_path + '/past_parameters.npy')[-1]).to(torch.float32)
    #mps_device = torch.device("mps")
    my_device = 'cpu'
    constant = read_constants(file1='cte_lambda.csv',file2='cst.csv',my_device = my_device)

    lr = 1e-3
    x_a = torch.ones(3)
    s_a_ = torch.eye(3)
    s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
    batch_size = data.len_data
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    s_a = s_a_*23
    model = Forward_Model(num_days=batch_size).to(my_device)
    initial_conditions(data,batch_size,model) #carefull with this step
    loss = custom_Loss(x_a,s_a,s_e,num_days=batch_size,my_device = my_device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    output = train_loop(next(iter(dataloader)),model,loss,optimizer,4000,kind='all',\
                        num_days=batch_size,constant = constant,perturbation_factors_ = perturbation_factors, scheduler = True)
    
    output_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/results_bayes_optimized'
    np.save(output_path + '/X_hat.npy',output['X_hat'])
    np.save(output_path + '/kd_hat.npy',output['kd_hat'])
    np.save(output_path + '/bbp_hat.npy',output['bbp_hat'])
    np.save(output_path + '/RRS_hat.npy',output['RRS_hat'])
    np.save(output_path + '/dates.npy',data.dates)
