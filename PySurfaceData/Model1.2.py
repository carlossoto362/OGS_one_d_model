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


#reading the initial value for the constants from two csv files, one with the constants dependent on lambda, and the other with
#the ones that do not depend on it.


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

constant = read_constants(file1='./cte_lambda.csv',file2='cst.csv')
#constant = read_constants(file1='IOP_T101/cte_lambda.csv',file2='cst.csv')
lambdas = np.array([412.5,442.5,490,510,555]).astype(float)

linear_regression=stats.linregress(lambdas,[constant['scattering_PH'][str(lamb)] for lamb in lambdas])
linear_regression_slope = linear_regression.slope
linear_regression_intercept = linear_regression.intercept

def read_data(data_path='./SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS',train=True):
    """
    function that reads the data stored in SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS/surface.yyy-mm-dd_12-00-00.txt
    read_data() returns a pandas DataFrame with all the data available on SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS.
    Each file has to be a csv with the columns
    lambda,RRS,E_dir,E_dif,zenith,PAR.
    
    """
    data = os.listdir(data_path)
    data.sort()
    if train == True:
        data = data[:int(len(data)*0.9)]
    else:
        data = data[int(len(data)*0.9):]
        
    names = ['lambda','RRS','E_dir','E_dif','zenith','PAR']
    all_data = pd.DataFrame(columns=['date','lambda','RRS','E_dir','E_dif','zenith','PAR'])
    for d in data:
        date = datetime.strptime(d,'surface.%Y-%m-%d_12-00-00.txt')
        one_data = pd.read_csv(data_path + '/' + d ,sep=' ',names=names)
        one_data['date'] = [date]*6
        all_data = pd.concat([all_data,one_data],ignore_index=True)
        
    return all_data.sort_values(by='date')


def read_initial_conditions(data,results_path='./results_first_run',lambdas=lambdas):
    """
    reads the results stored in results_path. This data is used as initial conditions for the learnings, to make
    the learning process for each day faster. The files are supposed to be csv files with columns
    ,412.5,442.5,490.0,510.0,555.0,chla,CDOM,NAP,loss

    This function is supposed to be used with a DataFrame of data and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = read_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = read_initial_conditions(data,results_path = './results_first_run')
    
    data is a pandas DataFrame with the columns
    date,lambda,RRS,E_dir,E_dif,zenith,PAR.
    
    read_result reads from all the files on the path, so make sure that the path has no other file than the results. Each result file has the data of one date, and has
    to be stored in a file named %Y-%m-%d.csv. Is meant to be used after storing the data from the save_results function. 
    
    """
    results_names = os.listdir(results_path)
    dates_results = [datetime.strptime(d,'%Y-%m-%d.csv') for d in results_names]
    results = data[data['date'].isin(dates_results)]
    results['RRS_MODEL']=np.empty(len(results))
    results['chla']=np.empty(len(results))
    results['NAP']=np.empty(len(results))
    results['CDOM']=np.empty(len(results))
    results['loss']=np.empty(len(results))
    dates_ = results['date'].unique()
    for i in range(len(dates_)):
        d = dates_[i]
        results_i = pd.read_csv(results_path + '/' +d.strftime('%Y-%m-%d.csv'))
        for lamb in lambdas:
            results.loc[(results['date']==d) & (results['lambda']==lamb),'RRS_MODEL'] = float(results_i[str(lamb)])

        results.loc[(results['date']==d),'chla'] = float(results_i['chla'])
        results.loc[(results['date']==d),'CDOM'] = float(results_i['CDOM'])
        results.loc[(results['date']==d),'NAP'] = float(results_i['NAP'])
        results.loc[(results['date']==d),'loss'] = float(results_i['loss'])

    return results

def read_kd_data(data,kd_data_path='messure_data/kd_data/BOUSSOLEFit_AntoineMethod_filtered_butterHighpass6days_min.csv'):
    """
    This function is supposed to be used with a DataFrame of data, and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = read_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = read_initial_conditions(data,results_path = './results_first_run')
    >>>data = read_kd_data(data)
    
    """
    kd_data = pd.read_csv(kd_data_path)
    kd_data['date'] = [datetime.strptime(kd_data['date'].iloc[k],'%Y-%m-%d') for k in range(len(kd_data))]
    result = data.merge(kd_data,on=['date','lambda'],how='left')
    return result

def read_chla_data(data,chl_data_path='messure_data/buoy.DPFF.2003-09-06_2012-12-31_999.dat'):
    """
    reads the chla data. The files are supposed to be csv files, including a column with YEAR, MONTH, DAY and chl. 

    This function is supposed to be used with a DataFrame of data and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = read_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = read_initial_conditions(data,results_path = './results_first_run')
    >>>data = read_kd_data(data)
    >>>data = read_chla_data(data)
    
    """
    buoy_data=pd.read_csv(chl_data_path,sep='\t')
    #print(buoy_data)
    
    buoy_results = np.empty((len(data)))
    for i in range(len(data)):
        date = data['date'].iloc[i]
        buoy_results[i] = buoy_data[ (buoy_data['YEAR'] == date.year) & (buoy_data['MONTH'] == date.month) & (buoy_data['DAY'] == date.day) ]['chl'].mean()
        data['buoy_chla'] = buoy_results
    return data

def read_bbp_data(data,bbp_data_path='messure_data/boussole_multi_rrs_bbpw_T10_IES20_2006-2012.csv'):
    """
    reads the bbp data. The files are supposed to be csv files, including a column with YEAR, MONTH, DAY and bbp_550,bbp_488 and bbp_442. 

    This function is supposed to be used with a DataFrame of data and a Dataframe of constants.
    Please create the global variables,
    
    >>>data = read_data(train=train)
    >>>data = data[data['lambda'].isin(lambdas)]
    >>>data = read_initial_conditions(data,results_path = './results_first_run')
    >>>data = read_kd_data(data)
    >>>data = read_chla_data(data)
    >>>data = read_bbp_data(data)
    
    """
    buoy_data = pd.read_csv(bbp_data_path,sep=';')
    data['buoy_bbp'] = np.ones(len(data))*np.nan
    for i in range(len(data)):
        if (data['lambda'].iloc[i] == 510) or (data['lambda'].iloc[i] == 412.5):
            pass
        elif data['lambda'].iloc[i] == 555:
            year = data['date'].iloc[i].year
            month = data['date'].iloc[i].month
            day = data['date'].iloc[i].day
            data['buoy_bbp'].iloc[i] = buoy_data[(buoy_data['YEAR'] == year) & (buoy_data['MONTH'] == month) & (buoy_data['DAY'] == day) ]['bbp_550'].mean()
        elif data['lambda'].iloc[i] == 490:
            year = data['date'].iloc[i].year
            month = data['date'].iloc[i].month
            day = data['date'].iloc[i].day
            data['buoy_bbp'].iloc[i] = buoy_data[(buoy_data['YEAR'] == year) & (buoy_data['MONTH'] == month) & (buoy_data['DAY'] == day) ]['bbp_488'].mean()
        elif data['lambda'].iloc[i] == 442.5:
            year = data['date'].iloc[i].year
            month = data['date'].iloc[i].month
            day = data['date'].iloc[i].day
            data['buoy_bbp'].iloc[i] = buoy_data[(buoy_data['YEAR'] == year) & (buoy_data['MONTH'] == month) & (buoy_data['DAY'] == day) ]['bbp_442'].mean()
    return data
        

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
    return constant['absortion_w'][str(lambda_.clone().detach().numpy())] + (constant['absortion_PH'][str(lambda_.clone().detach().numpy())] * perturbation_factors['mul_factor_a_ph'])*chla + \
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
    return constant['scattering_w'][str(lambda_.clone().detach().numpy())] + scattering_ph(lambda_,perturbation_factors,tensor=tensor) * Carbon(chla,PAR,perturbation_factors,tensor=tensor) + \
        scattering_NAP(lambda_,tensor=tensor) * NAP

#################Functions for the backscattering coefitient#############

def backscattering(lambda_,PAR,chla,NAP,perturbation_factors,tensor=True):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. The variable "tensor" specify if the input and output of the function is a torch.tensor. 
    """
    return constant['backscattering_w'][str(lambda_.clone().detach().numpy())] + perturbation_factors['mul_factor_backscattering_ph']*constant['backscattering_PH'][str(lambda_.clone().detach().numpy())] * \
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
        return constant['backscattering_PH'][str(lambda_.clone().detach().numpy())] * perturbation_factors['mul_factor_backscattering_ph'] * \
        Carbon(chla,PAR,perturbation_factors,tensor=tensor) + perturbation_factors['mul_factor_backscattering_nap']*0.005 * scattering_NAP(lambda_,tensor=tensor) * NAP
    else:
        return constant['backscattering_PH'][str(lambda_.clone().detach().numpy())] * perturbation_factors['mul_factor_backscattering_ph'] * \
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




#################################Putting all together#################################################
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

    def forward(self,x_data,perturbation_factors):
        """
        x_data: pandas dataframe with columns [E_dif,E_dir,lambda,zenith,PAR].
        """
        Rrs = torch.empty(5)
        for i in range(5):
            #Rrs[i,] = Rrs_MODEL(x_data['E_dif'].iloc[i],x_data['E_dir'].iloc[i],x_data['lambda'].iloc[i],\
            #x_data['zenith'].iloc[i],x_data['PAR'].iloc[i],self.chla,self.NAP,self.CDOM,perturbation_factors)
            Rrs[i,] = Rrs_MODEL(x_data[i,0],x_data[i,1],x_data[i,2],\
            x_data[i,3],x_data[i,4],self.chla,self.NAP,self.CDOM,perturbation_factors)
        return Rrs


    
def train_loop(data_i,model,loss_fn,optimizer,N,perturbation_factors):
    """
    The train loop evaluates the Remote Sensing Reflectance RRS for each wavelength>>>pred=model(data_i), evaluates the loss function
    >>>loss=loss_fn(pred,y), force the value of the parameters (chla,NAP,CDOM) to be positive, evaluates the gradient of RRS with respect
    to the parameters, >>>loss.backward(), modifies the value of the parameters according to the optimizer criterium, >>>optimizer.step(),
    sets the gradient of RRS to cero, and prints the loss for a given number of iterations. This procedure is performed N times or untyl a treshold is ashieved. 
    After N iterations, it returns two lists with the evolution of the loss function and the last evaluation of the model. 
    """
    size = 5
    ls_val=[]
    ls_count=[]

    criterium = 1
    criterium_2 = 0
    i=0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #file_ = open('results_imprubment.txt','w')
    while (((criterium >1e-12 ) & (i<N)) or ((criterium_2 < 100)&(i<N))):
        y = data_i[:,5]
        pred = model(data_i[:,:5],perturbation_factors)
        loss = loss_fn(pred,y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for p in model.parameters():
            p.data.clamp_(0)
        
        ls_val.append(loss.item())
        ls_count.append(i)
        #print(i,list(model.parameters()))
        if i != 0:
            criterium = ls_val[-2] - ls_val[-1]
        if criterium <=1e-12:
            criterium_2+=1
        i+=1
        scheduler.step(loss)
    return ls_val,ls_count,pred

def run_inversion(x_data_tensor_T,perturbation_factors,rank,world_size,initial_conditions_path=None,number_iterations_N = 100):
    """
    Designed to make multiple inversion problems in parallel manner. Because of conflicts with pytorch, I end up dont making it on parallen manually.

	It runs the inversion problem for N diferent days.

	x_data_tensor_T is a 26*N torch.tensor, with each column representing the quantities:
	col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26
        RRS412,RRS442,RRS490,RRS510,RRS555,Edif412,Edif442,Edif490,Edif510,Edif555,Edir412,Edir442,Edir490,Edir510,Edir555,
        lambda412,lambda442,lambda490,lambda510,lambda555,zenith,PAR,chla,nap,cdom,date

	perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
	of some of the constants of the bio quimical model.

	The output is a 9*N torch.tensor, with columns representing
	col1,col2,col3,col4,col5,col6,col7,col8,col9
        chla,kd412,kd442,kd490,kd510,kd555,bbp442,bbp490,bbp555
    """

    boss = 0
    num_workers = world_size - 1
    
    x_data_size = x_data_tensor_T.size()[0]
    x_result = torch.empty(x_data_size,9)
    #x_result_workers = torch.empty(num_workers,9)
    #pool = mp.Pool(cores)
    #input = []
    job = 0
    #dist.barrier()
    """
    while job < x_data_size:
        for worker_iter in range(1,num_workers+1):
            if rank == worker_iter:
                x_result_workers[worker_iter-1] = run_one_inversion([job,x_data_tensor_T[job],perturbation_factors,x_result])[1]
                dist.send(tensor=x_result_workers[worker_iter-1], dst=boss)
                #print("sending job {} from worker {}.".format(job+1,rank))
            job+=1
            if job == x_data_size:
                break
        last_worker = worker_iter
        if rank == boss:
            for worker_iter in range(1,last_worker + 1):
                dist.recv(tensor=x_result_workers[worker_iter-1],src=worker_iter)
                #print("receive job {} from worker {}, last job in process is {}.".format(job-last_worker + worker_iter,worker_iter,job))
                x_result[job-last_worker + worker_iter-1] = x_result_workers[worker_iter-1]
    """

    """
    last_job = 0
    x_result_fut = []
    while job < x_data_size:
        for worker_iter in range(1,num_workers+1):
            input_ = [job,x_data_tensor_T[last_job + worker_iter - 1],perturbation_factors]
            x_result_fut.append(dist.rpc.rpc_async(worker_iter, run_one_inversion, args=(input_),timeout=60*10) )
            job+=1
            if job == x_data_size:
                break
        last_job = job

    for xr in range(len(x_result_fut)):
        x_result[xr] = x_result_fut[xr].wait()
    print(x_result[0])
    #dist.barrier()
    """
    time1 = time.time()
    job = 0
    for x_data_tensor in x_data_tensor_T:
        x_result[job] = run_one_inversion(job,x_data_tensor,perturbation_factors,initial_conditions_path=initial_conditions_path,number_iterations_N = number_iterations_N)
        job+=1
    return x_result




    #if rank == boss:
    #    return x_result


def run_one_inversion(x_index,x_data_tensor,perturbation_factors,initial_conditions_path=None,number_iterations_N = 100):
        """
	Runs the inversion problem for the bio-optical model, using the loss_function MSELoss(), and the optimizer torch.optim.Adam.

	This function is supposed to be used with a DataFrame of data and a Dataframe of constants.

	input is an iterable, with three elements, the first is an index, in order preserved the ordering of the elements, in case is needed. 
	the second is x_data_tensor, descrived in the function run_inversion, and the last one is the perturbation_factors, also described in
	run_inversion.

        """

        #x_data_tensor = input[1]
        #x_index = input[0]
        #perturbation_factors = input[2]
        timeInit=time.time()
        x_data = torch.empty((5,7))
        
        
        dates_ = []
        lambdas = []
        RRS=[]
        Edif=[]
        Edir=[]
        zenithh=[]
        PAR=[]

        for lam in range(5):
            x_data[lam,2] = x_data_tensor[15+lam] # lambda
            x_data[lam,5] = x_data_tensor[lam] # RRS
            x_data[lam,0] = x_data_tensor[5+lam] # Edif
            x_data[lam,1] = x_data_tensor[10+lam] # Edir
            x_data[lam,3] = x_data_tensor[20] # zenithh
            x_data[lam,4] = x_data_tensor[21] # PAR

        x_data = x_data[x_data.sort(dim=0,descending=True)[1][:,2]] #descending: from biguer to smallest, second column is the indexes, then I select the column of lambdas, equivalent to
        #what before was "x_data[:,5] = x_data.sort_values(by=['lambda'],ascending=False)" with pandas ordering. 

        N = number_iterations_N
        model = MODEL().to("cpu")
        state_dict = model.state_dict()
        state_dict['chparam'] = torch.ones((3), dtype=torch.float32)*x_data_tensor[22:25]#initial conditions equal to the first run (with the base parameters)

        model.load_state_dict(state_dict)
    
        learning_rate = 5e-3 #this is to use gradient descent. 
        loss_function = nn.MSELoss() #MSE, the same used by Paolo.
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        #optimizer = torch.optim.LBFGS(model.parameters(),lr=learning_rate) #this requires the closure function, but is much slower.
        ls_val,ls_count,pred = train_loop(x_data,model,loss_function,optimizer,N,perturbation_factors)

        for par in model.parameters():
            par.require_grad=False
        chla = list(model.parameters())[0][0]
        nap = list(model.parameters())[0][1]
        cdom = list(model.parameters())[0][2]
        if initial_conditions_path == None:
            pass
        else:
            np.save(initial_conditions_path+'/'+str(int(x_data_tensor[-1].clone().detach().numpy())),\
                    np.array([chla.clone().detach().numpy(),nap.clone().detach().numpy(),cdom.clone().detach().numpy()]),allow_pickle=False)

        x_result_i = torch.empty(9)
        x_result_i[0] = chla

        for i in range(5): #I ordered the data as 412.5,445.5,490,510,555 so...

            x_result_i[i+1] =  kd(9.,x_data[i,0],x_data[i,1],x_data[i,2],x_data[i,3],\
                            x_data[i,4],chla,nap,cdom,perturbation_factors)

            if i == 1:
                bbp_index = 6
            elif i == 2:
                bbp_index = 7
            elif i == 4:
                bbp_index = 8
            else:
                continue
            
            x_result_i[bbp_index] = bbp(x_data[i,0],x_data[i,1],x_data[i,2],x_data[i,3],\
                                    x_data[i,4],chla,nap,cdom,perturbation_factors)
            
        return x_result_i



class customTensorData():
    """
	Custom class to load the data and transform it into tensors that are meant to be used with the DataLoader function of pytorch. 
    """
    def __init__(self, initial_conditions_path,data_path,which='Train', transform=None, target_transform=None,rank=0,initial_conditions_path_created=False,past_initial_conditions_path_index=0):
        
        x_train = np.load(data_path + '/x_train.npy',allow_pickle=False)
        y_train = np.load(data_path + '/y_train.npy',allow_pickle=False)
        x_test = np.load(data_path + '/x_test.npy',allow_pickle=False)
        y_test = np.load(data_path + '/y_test.npy',allow_pickle=False)
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
            
        initial_conditions_path_index = torch.tensor(0) + past_initial_conditions_path_index
        if rank == 0:
            initial_conditions_path_iter = initial_conditions_path + '_' + str(initial_conditions_path_index.numpy())
            if initial_conditions_path_created == False:
                while initial_conditions_path_created == False:
                    if os.path.exists(initial_conditions_path_iter):
                        initial_conditions_path_index = initial_conditions_path_index + 1
                        initial_conditions_path_iter = initial_conditions_path + '_' + str(initial_conditions_path_index.numpy())
                    else:
                        os.mkdir(initial_conditions_path_iter)
                        initial_conditions_path_created = True
        else:
            pass
        
        dist.monitored_barrier(timeout=timedelta(seconds=1))
        dist.broadcast( tensor = initial_conditions_path_index,src=0 )
        self.initial_conditions_path = initial_conditions_path + '_' + str(initial_conditions_path_index.numpy())
        self.initial_conditions_path_index = initial_conditions_path_index
        
        #saving the initial conditions in (probably temporary) files
        if rank == 0:
            for image in self.images:
                np.save(self.initial_conditions_path+'/'+str(int(image[-1])),image[22:25],allow_pickle=False)

        self.images = np.delete(self.images,[22,23,24],axis=1)#the initial conditions are suposed to be changing after each iteration. 
        
        self.labels_stds = np.nanstd(self.labels,axis=0)
        self.images_stds = np.nanstd(self.images,axis=0)
        self.labels_means = np.nanmean(self.labels,axis=0)
        self.images_means = np.nanmean(self.images,axis=0)
        self.images_columns = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555','Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','lambda_412','lambda_442',\
                               'lambda_490','lambda_510','lambda_555','zenith','PAR','date']
        self.info = '''the columns 23,24 and 25 are the initial conditions for the learning, not part of the input data,
        date is also not part of the image, but aan index, indicating the number of days since the first of january of 2000.'''

        self.data_path = data_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Now that im transforming the data in tensors, I'm going to loos the names, so the order is important
	col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26
        RRS412,RRS442,RRS490,RRS510,RRS555,Edif412,Edif442,Edif490,Edif510,Edif555,Edir412,Edir442,Edir490,Edir510,Edir555,
        lambda412,lambda442,lambda490,lambda510,lambda555,zenith,PAR,chla,nap,cdom,date

        and for the labels,
	col1,col2,col3,col4,col5,col6,col7,col8,col9
        chla,kd412,kd442,kd490,kd510,kd555,bbp442,bbp490,bbp555
        """
        
        label = torch.tensor(self.labels[idx])

        #reading the initial conditions:
        initial_conditions = np.load(self.initial_conditions_path + '/'+ str(int(self.images[idx][-1]))+'.npy',allow_pickle=False)

        image = np.append( self.images[idx][:-1],initial_conditions )
        image = np.append(image,self.images[idx][-1])
        image = torch.tensor(image)
          
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class Parameter_Estimator(nn.Module):
    """
	Model that attempts to learn the perturbation factors. 
    """
    def __init__(self):
        super().__init__()
        self.mul_factors = nn.Parameter(torch.ones(13, dtype=torch.float32), requires_grad=True)

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

    def forward(self,image,rank,world_size,initial_conditions_path=None,number_iterations_N = 100):
        """
        x_data: pandas dataframe with columns [E_dif,E_dir,lambda,zenith,PAR].
        """

        return run_inversion(image,self.perturbation_factors,rank,world_size,initial_conditions_path=initial_conditions_path,number_iterations_N = number_iterations_N)

def custom_LSELoss(input,output,stds):
	"""
	My data has some nan on it, so this function returns the least square error loss function, taking into consideration the nan elements.
	"""
	custom_array = ((output-input)/stds)**2
	lens = torch.tensor([len(element[element!=0]) for element in output])
	means_output = custom_array.sum(axis=1)/lens
	return means_output.sum()
    

class WeightClipper(object):
    """
	Object that clamps the parameters between 0 and 2. 
    """
    def __init__(self):
        pass

    def __call__(self, module):
        # filter the variables to get the ones you want
        with torch.no_grad():
            for param in module.parameters():
                param.clamp_(0.01, 1.99)



def train_loop_parameters(dataloader, model_par, loss_fn_par, optimizer_par,scheduler_par,labels_stds,rank,world_size,batch_size,learning_rate,initial_conditions_path=None,num_iteration=1):
    """
	train loop for the learning of the perturbation factors. It has an scheduler in order to increase the velocity of convergence, and a cliper, to set 
	a constrains in the parameters. 
    """
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    job_ = 0
    last_job = 0
    len_batches = len(dataloader)
    torch.manual_seed(1762+num_iteration)
    batches = list(enumerate(dataloader))
    while job_ < len_batches:
        if rank == 0:
            training_parameters_file = open('training_parameters_'+str(batch_size)+'_'+str(learning_rate)+'.csv','a')
        
        if rank == 0:
            print('starting batch',job_,'to',job_+world_size,'from',len_batches)
            batch_init_time = time.time()

        for work_iter in range(world_size):
            if rank == work_iter:
                batch = batches[last_job + work_iter][0]
                X = batches[last_job + work_iter][1][0]
                y_nan = batches[last_job + work_iter][1][1]
                y = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
                time2 = time.time()
                pred = model_par(X,rank,world_size,initial_conditions_path=initial_conditions_path)
                pred = torch.masked_fill(pred,torch.isnan(y_nan),0)
                loss = loss_fn_par(pred,y,labels_stds)
                loss.backward()
            job_ += 1
            if job_ == len_batches:
                break
        last_worker = work_iter
        last_job = job_
        for i in range(len(list(model_par.parameters()))):
            dist.all_reduce(tensor=list(model_par.parameters())[i].grad, op = dist.ReduceOp.SUM)
            list(model_par.parameters())[i].grad /= batch_size*(last_worker+1)
        dist.all_reduce(loss,op = dist.ReduceOp.SUM)
        loss /= batch_size*(last_worker+1)
        
        optimizer_par.step()
        cliper = WeightClipper()
        model_par.apply(cliper)
        scheduler_par.step(loss)
        for i in range(len(list(model_par.parameters()))):
            dist.broadcast( tensor = list(model_par.parameters())[i],src=0 ) # just to be shure that all the workers have the same parameters.
        dist.barrier()
        if rank == 0:
            training_parameters_file.write('# '+ str(job_) + ' to ' + str(job_+world_size) + ' from ' + str(len_batches) + ', time used, ' + str(time.time() - batch_init_time) +  '\n')
            for i in range(len(list(model_par.parameters())[0])):
                num_print = list(model_par.parameters())[0][i].clone().detach().numpy()
                training_parameters_file.write(str(num_print) + ',')
            training_parameters_file.write(str(loss.clone().detach().numpy())+'\n')
            print('time used for',last_worker,'batches', time.time() - batch_init_time)
            training_parameters_file.close()

        optimizer_par.zero_grad()
        


def test_loop_parameters(dataloader, model_par, loss_fn_par,labels_stds,rank,world_size,batch_size,learning_rate):
    """
	test loop, that evaluates the model with the dataloader. 
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    dist.barrier()#training until all the cores are in this point
    if rank == 0:
        print('starting test...')
        batch_init_time = time.time()
    model_par.eval()
    test_loss = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    for param in model_par.parameters():
        param.requires_grad = False

    num_batch = 0
    len_batches = len(dataloader)
    batches = list(enumerate(dataloader))
    job_ = 0
    last_job = 0
    while job_ < len_batches:
        
        dist.barrier()
        loss = torch.tensor(0)
        for work_iter in range(world_size):
            if rank == work_iter:
                batch = batches[last_job + work_iter][0]
                X = batches[last_job + work_iter][1][0]
                y_nan = batches[last_job + work_iter][1][1]
                y = torch.masked_fill(y_nan,torch.isnan(y_nan),0)
                pred = model_par(X,rank,world_size,number_iterations_N = 4000)
                pred = torch.masked_fill(pred,torch.isnan(y_nan),0)
                loss = loss_fn_par(pred,y,labels_stds)
            job_ += 1
            if job_ == len_batches:
                break
        last_worker = work_iter
        last_job = job_
        dist.all_reduce(tensor=loss, op = dist.ReduceOp.SUM)
        test_loss += loss
        dist.barrier()
    test_loss = test_loss/ ( len_batches*batch_size )
    for param in model_par.parameters():
        param.requires_grad = True
    if rank == 0:
        training_parameters_file = open('training_parameters_test_'+str(batch_size)+'_'+str(learning_rate)+'.csv','a')
        training_parameters_file.write("Avg loss: {:.8f},".format(test_loss.clone().detach().numpy()))
        for i in range(len(list(model_par.parameters())[0])):
            num_print = list(model_par.parameters())[0][i].clone().detach().numpy()
            if i != len(list(model_par.parameters())[0])-1:
                end = ','
            else:
                end = '\n'
            training_parameters_file.write(str(num_print) + end)
        training_parameters_file.close()
        print('time used for test:' ,time.time() - batch_init_time)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size,timeout = timedelta(seconds=60*30*100))
    if rank == 0:
        name = 'boss'
    else:
        name = 'worker{}'.format(rank)
    #dist.rpc.init_rpc(name, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    #dist.rpc.shutdown()
    

    

def parallel_training(rank,world_size,batch_size,learning_rate):
    
    warnings.filterwarnings('ignore')
    pd.options.mode.chained_assignment = None
    boss = 0
    setup(rank,world_size)
    if rank == boss:
        time_zero=time.time()
    
    device = (
	"cuda"
	if torch.cuda.is_available()
	else "cpu"
    )
    
    initial_conditions_path = '/Users/carlos/Documents/surface_data_analisis/npy_data/'
    data_path = '/Users/carlos/Documents/surface_data_analisis/npy_data/'
    train_data = customTensorData(initial_conditions_path,data_path,which='Train',rank=rank)
    labels_stds = torch.tensor(train_data.labels_stds)
    test_data = customTensorData(initial_conditions_path,data_path,which='Test',rank=rank,initial_conditions_path_created=True,past_initial_conditions_path_index=train_data.initial_conditions_path_index)
    #labels_means = test_data.get_means()[1]
    
    learning_rate = float(learning_rate)
    batch_size = int(batch_size)
    epochs = 20

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  #next_iteration_image,next_iteration_label = next(iter(train_dataloader))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False) 
    #train_dataloader = test_dataloader
    model_par = Parameter_Estimator()
    #model_par = DDP(model_par, device_ids=[rank])
    loss_fn_par = custom_LSELoss
    optimizer_par = torch.optim.Adam(model_par.parameters(), lr=learning_rate)
    dist.barrier()    
    if rank == boss:
        print('time to load data and the model: ', time.time() - time_zero)
        training_time = time.time()
    scheduler_par = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_par, 'min')
    test_loop_parameters(test_dataloader, model_par, loss_fn_par,labels_stds,rank,world_size,batch_size,learning_rate)
    
    for t in range(epochs):
        if rank == 0:
            print("Epoch: ",t)
        train_loop_parameters(train_dataloader,model_par,loss_fn_par,optimizer_par,scheduler_par,labels_stds,rank,world_size,batch_size,learning_rate,\
                              initial_conditions_path = train_data.initial_conditions_path,num_iteration=t)
        test_loop_parameters(test_dataloader, model_par, loss_fn_par,labels_stds,rank,world_size,batch_size,learning_rate)
    if rank == boss:
        print('total time: '+ str(time.time() - time_zero))
    cleanup()
    

    
def run_demo(demo_fn, world_size,batch_size,learning_rate):
    processes = []
    try:
        for rank in range(world_size):
            p = mp.Process(target=demo_fn, args=(rank,world_size,batch_size,learning_rate))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for rank in range(world_size):
            print('Terminating Process ',rank)
            p.terminate()
            

if __name__ == '__main__':
    if len(sys.argv)>1:
        minibatch_size = sys.argv[1]
        learning_rate = sys.argv[2]
    else:
        minibatch_size = 1
        learning_rate = 0.0005
    for batch_size in [minibatch_size]:
        run_demo(parallel_training, 1,batch_size,learning_rate)

        

