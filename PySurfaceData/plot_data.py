#!/usr/bin/env python

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

from ModelOnePointOne import *

import warnings




def read_train_data(initial_conditions_path,data_path,first_run = False):
    lambdas = np.array([412.5,442.5,490,510,555]).astype(float)
    if first_run == True:
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
        data = read_data(train=False,all_=True)
        data = data[data['lambda'].isin(lambdas)]
        data = read_initial_conditions(data,results_path = initial_conditions_path)
        data = read_kd_data(data)
        data = read_chla_data(data)
        data = read_bbp_data(data)
        data['kd'] = [kd(9,data['E_dif'].iloc[i],data['E_dir'].iloc[i],data['lambda'].iloc[i],\
                     data['zenith'].iloc[i],data['PAR'].iloc[i],data['chla'].iloc[i],data['NAP'].iloc[i],data['CDOM'].iloc[i],\
                     initial_perturbation_factors,tensor=False) for i in range(len(data))]
        
    
        data['bbp'] = [bbp(data['E_dif'].iloc[i],data['E_dir'].iloc[i],data['lambda'].iloc[i],\
                     data['zenith'].iloc[i],data['PAR'].iloc[i],data['chla'].iloc[i],data['NAP'].iloc[i],data['CDOM'].iloc[i],\
                     initial_perturbation_factors,tensor=False) for i in range(len(data))]
        data = data.sort_values(by=['date','lambda'])
    else:

        data = read_data(data_path,train=False,all_=True)
        data = data[data['lambda'].isin(lambdas)]
        data = read_kd_data(train_data)
        data = read_chla_data(train_data)
        data = read_bbp_data(train_data)
        data = data.sort_values(by=['date','lambda'])
    return data


def read_second_run(second_run_path,include_uncertainty=False,optimized=True):
    dates = []
    if include_uncertainty == False:
        second_run_output = pd.DataFrame(columns=['RRS_o_555','RRS_o_510','RRS_o_490','RRS_o_442','RRS_o_412',\
                                          'chla_o','NAP_o','CDOM_o','kd_o_555','kd_o_510','kd_o_490','kd_o_442',\
                                          'kd_o_412','bbp_o_555','bbp_o_490','bbp_o_442'])





        for file_ in os.listdir(second_run_path):
            if file_ == 'README':
                pass
            else:
                np_file = np.load(second_run_path + '/' + file_)
                dates.append(datetime.strptime(file_,'output_%Y%m%d.npy'))
                second_run_output.loc[len(second_run_output)] = np_file
    


    elif include_uncertainty == True:
        second_run_output = pd.DataFrame(columns=['RRS_o_412','RRS_o_442','RRS_o_490','RRS_o_510','RRS_o_555',\
                                                  'chla_o', 'delta_chla_o','NAP_o','delta_NAP_o','CDOM_o','delta_CDOM_o',\
                                                  'kd_o_412','delta_kd_o_412','kd_o_442','delta_kd_o_442','kd_o_490','delta_kd_o_490','kd_o_510','delta_kd_o_510',\
                                                  'kd_o_555','delta_kd_o_555','bbp_o_442','delta_bbp_o_442','bbp_o_490','delta_bbp_o_490','bbp_o_555','delta_bbp_o_555'])
        if optimized == False:
            for file_ in os.listdir(second_run_path):
                if file_ == 'README':
                    pass
                else:
                    dates.append(datetime.strptime(file_,'%Y%m%d'))
                    column = np.empty(27)
                    Y_hat = np.load(second_run_path + '/' + file_ + '/Y_hat.npy')
                    X_hat = np.load(second_run_path + '/' + file_ + '/X_hat.npy')
                    S_hat = np.load(second_run_path + '/' + file_ + '/S_hat.npy')
                    kd_hat = np.load(second_run_path + '/' + file_ + '/kd_hat.npy')
                    bbp_hat = np.load(second_run_path + '/' + file_ + '/bbp_hat.npy')
                    column[:5] = Y_hat
                    column[5] = X_hat[0]
                    column[6] = S_hat[0][0]**(1/2)
                    column[7] = X_hat[1]
                    column[8] = S_hat[1][1]**(1/2)
                    column[9] = X_hat[2]
                    column[10] = S_hat[2][2]**(1/2)
                    column[11:21] = kd_hat
                    column[21:27] = bbp_hat
                    second_run_output.loc[len(second_run_output)] = column
                
        else:
            second_run_output[second_run_output.columns[:5]] = np.load(second_run_path + '/RRS_hat.npy')
            second_run_output[second_run_output.columns[5:11]] = np.load(second_run_path + '/X_hat.npy')
            second_run_output[second_run_output.columns[11:21]] = np.load(second_run_path + '/kd_hat.npy')
            second_run_output[second_run_output.columns[21:]] = np.load(second_run_path + '/bbp_hat.npy')
            dates = np.load(second_run_path + '/dates.npy')
            dates = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in dates]
        
        
    else:
        print('include_uncertainty has to be True or False')
        
                
    second_run_output['date'] = dates
    second_run_output.sort_values(by='date',inplace=True)

    return second_run_output



def add_messure_data(second_run_output,train_data,include_first_run=True):
    
    if include_first_run == True:
        second_run_output = second_run_output.merge(train_data[train_data['lambda']==555]['date'],how='right',on='date')
        second_run_output['chla_d'] = train_data[train_data['lambda']==555]['buoy_chla'].to_numpy()
        second_run_output['chla_l'] = train_data[train_data['lambda']==555]['chla'].to_numpy()

        second_run_output['NAP_l'] = train_data[train_data['lambda']==555]['NAP'].to_numpy()
        second_run_output['CDOM_l'] = train_data[train_data['lambda']==555]['CDOM'].to_numpy()

    
        second_run_output['kd_d_555'] = train_data[train_data['lambda']==555]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_510'] = train_data[train_data['lambda']==510]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_490'] = train_data[train_data['lambda']==490]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_442'] = train_data[train_data['lambda']==442.5]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_412'] = train_data[train_data['lambda']==412.5]['kd_filtered_min'].to_numpy()

        second_run_output['kd_l_555'] = train_data[train_data['lambda']==555]['kd'].to_numpy()
        second_run_output['kd_l_510'] = train_data[train_data['lambda']==510]['kd'].to_numpy()
        second_run_output['kd_l_490'] = train_data[train_data['lambda']==490]['kd'].to_numpy()
        second_run_output['kd_l_442'] = train_data[train_data['lambda']==442.5]['kd'].to_numpy()
        second_run_output['kd_l_412'] = train_data[train_data['lambda']==412.5]['kd'].to_numpy()

        second_run_output['bbp_d_555'] = train_data[train_data['lambda']==555]['buoy_bbp'].to_numpy()
        second_run_output['bbp_d_490'] = train_data[train_data['lambda']==490]['buoy_bbp'].to_numpy()
        second_run_output['bbp_d_442'] = train_data[train_data['lambda']==442.5]['buoy_bbp'].to_numpy()

        second_run_output['bbp_l_555'] = train_data[train_data['lambda']==555]['bbp'].to_numpy()
        second_run_output['bbp_l_490'] = train_data[train_data['lambda']==490]['bbp'].to_numpy()
        second_run_output['bbp_l_442'] = train_data[train_data['lambda']==442.5]['bbp'].to_numpy()

        second_run_output['rrs_d_555'] = train_data[train_data['lambda']==555]['RRS'].to_numpy()
        second_run_output['rrs_d_510'] = train_data[train_data['lambda']==510]['RRS'].to_numpy()
        second_run_output['rrs_d_490'] = train_data[train_data['lambda']==490]['RRS'].to_numpy()
        second_run_output['rrs_d_442'] = train_data[train_data['lambda']==442.5]['RRS'].to_numpy()
        second_run_output['rrs_d_412'] = train_data[train_data['lambda']==412.5]['RRS'].to_numpy()

        second_run_output['rrs_l_555'] = train_data[train_data['lambda']==555]['RRS_MODEL'].to_numpy()
        second_run_output['rrs_l_510'] = train_data[train_data['lambda']==510]['RRS_MODEL'].to_numpy()
        second_run_output['rrs_l_490'] = train_data[train_data['lambda']==490]['RRS_MODEL'].to_numpy()
        second_run_output['rrs_l_442'] = train_data[train_data['lambda']==442.5]['RRS_MODEL'].to_numpy()
        second_run_output['rrs_l_412'] = train_data[train_data['lambda']==412.5]['RRS_MODEL'].to_numpy()
       
    else:
        second_run_output = second_run_output.merge(train_data[train_data['lambda']==555]['date'],how='right',on='date')
        second_run_output['chla_d'] = train_data[train_data['lambda']==555]['buoy_chla'].to_numpy()
        
        second_run_output['kd_d_555'] = train_data[train_data['lambda']==555]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_510'] = train_data[train_data['lambda']==510]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_490'] = train_data[train_data['lambda']==490]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_442'] = train_data[train_data['lambda']==442.5]['kd_filtered_min'].to_numpy()
        second_run_output['kd_d_412'] = train_data[train_data['lambda']==412.5]['kd_filtered_min'].to_numpy()

        second_run_output['bbp_d_555'] = train_data[train_data['lambda']==555]['buoy_bbp'].to_numpy()
        second_run_output['bbp_d_490'] = train_data[train_data['lambda']==490]['buoy_bbp'].to_numpy()
        second_run_output['bbp_d_442'] = train_data[train_data['lambda']==442.5]['buoy_bbp'].to_numpy()

        
        second_run_output['rrs_d_555'] = train_data[train_data['lambda']==555]['RRS'].to_numpy() 
        second_run_output['rrs_d_510'] = train_data[train_data['lambda']==510]['RRS'].to_numpy() 
        second_run_output['rrs_d_490'] = train_data[train_data['lambda']==490]['RRS'].to_numpy()
        second_run_output['rrs_d_442'] = train_data[train_data['lambda']==442.5]['RRS'].to_numpy() 
        second_run_output['rrs_d_412'] = train_data[train_data['lambda']==412.5]['RRS'].to_numpy()
        
    return second_run_output

                
def compute_error(second_run_output,first_run=True):
    if first_run == True:
        first_run_error = pd.DataFrame({'chla_e':(second_run_output['chla_d'] - second_run_output['chla_l'])**2/(second_run_output['chla_d'].std())**2})

        first_run_error['kd_e_555'] = (second_run_output['kd_d_555'] - second_run_output['kd_l_555'])**2/(second_run_output['kd_d_555'].std())**2
        first_run_error['kd_e_510'] = (second_run_output['kd_d_510'] - second_run_output['kd_l_510'])**2/(second_run_output['kd_d_510'].std())**2
        first_run_error['kd_e_490'] = (second_run_output['kd_d_490'] - second_run_output['kd_l_490'])**2/(second_run_output['kd_d_490'].std())**2
        first_run_error['kd_e_442'] = (second_run_output['kd_d_442'] - second_run_output['kd_l_442'])**2/(second_run_output['kd_d_442'].std())**2
        first_run_error['kd_e_412'] = (second_run_output['kd_d_412'] - second_run_output['kd_l_412'])**2/(second_run_output['kd_d_412'].std())**2

        first_run_error['bbp_e_555'] = (second_run_output['bbp_d_555'] - second_run_output['bbp_l_555'])**2/second_run_output['bbp_d_555'].std()**2
        first_run_error['bbp_e_490'] = (second_run_output['bbp_d_490'] - second_run_output['bbp_l_490'])**2/second_run_output['bbp_d_490'].std()**2
        first_run_error['bbp_e_442'] = (second_run_output['bbp_d_442'] - second_run_output['bbp_l_442'])**2/second_run_output['bbp_d_442'].std()**2

        first_run_error_mean = first_run_error.mean(axis=1)
        first_run_error_mean = first_run_error_mean.mean()
        print('first run error: ',first_run_error_mean)
    else:
        pass
    
    second_run_error = pd.DataFrame({'chla_e':(second_run_output['chla_d'] - second_run_output['chla_o'])**2/(second_run_output['chla_d'].std())**2})

    second_run_error['kd_e_555'] = (second_run_output['kd_d_555'] - second_run_output['kd_o_555'])**2/(second_run_output['kd_d_555'].std())**2
    second_run_error['kd_e_510'] = (second_run_output['kd_d_510'] - second_run_output['kd_o_510'])**2/(second_run_output['kd_d_510'].std())**2
    second_run_error['kd_e_490'] = (second_run_output['kd_d_490'] - second_run_output['kd_o_490'])**2/(second_run_output['kd_d_490'].std())**2
    second_run_error['kd_e_442'] = (second_run_output['kd_d_442'] - second_run_output['kd_o_442'])**2/(second_run_output['kd_d_442'].std())**2
    second_run_error['kd_e_412'] = (second_run_output['kd_d_412'] - second_run_output['kd_o_412'])**2/(second_run_output['kd_d_412'].std())**2

    second_run_error['bbp_e_555'] = (second_run_output['bbp_d_555'] - second_run_output['bbp_o_555'])**2/(second_run_output['bbp_d_555'].std())**2
    second_run_error['bbp_e_490'] = (second_run_output['bbp_d_490'] - second_run_output['bbp_o_490'])**2/(second_run_output['bbp_d_490'].std())**2
    second_run_error['bbp_e_442'] = (second_run_output['bbp_d_442'] - second_run_output['bbp_o_442'])**2/(second_run_output['bbp_d_442'].std())**2

    second_run_error_mean = second_run_error.mean(axis=1).pow(1./2.)
    second_run_error_mean = second_run_error_mean.mean()
    print('Second run error: ',second_run_error_mean)


def plot_RRS_data(second_run_output):
    start,end=0,5
    y_labels = ['$R_{RS,\lambda=412.5}$ ($sr^{-1}$)','$R_{RS,\lambda=442.5}$ ($sr^{-1}$)','$R_{RS,\lambda=490}$ ($sr^{-1}$)','$R_{RS,\lambda=510}$ ($sr^{-1}$)','$R_{RS,\lambda=555}$ ($sr^{-1}$)']
    dates = second_run_output['date']
    plot_d_data = [second_run_output['rrs_d_412'],second_run_output['rrs_d_442'],second_run_output['rrs_d_490'],second_run_output['rrs_d_510'],second_run_output['rrs_d_555']]
    plot_o_data = [second_run_output['RRS_o_412'],second_run_output['RRS_o_442'],second_run_output['RRS_o_490'],second_run_output['RRS_o_510'],second_run_output['RRS_o_555']]
    corrections =  [0.2,0.25,0.35,0.25,0]
    
    fig, ax = plt.subplots(end-start, 2,width_ratios = [2.5,1],figsize=(15,9))
    for i in range(start,end):

        ks = stats.kstest(plot_d_data[i][~(plot_d_data[i].isnull()).to_numpy()],plot_o_data[i][~(plot_d_data[i].isnull()).to_numpy()])
        corr = stats.pearsonr(plot_d_data[i][~plot_d_data[i].isnull().to_numpy()][~plot_o_data[i][~plot_d_data[i].isnull().to_numpy()].isnull().to_numpy()],\
                              plot_o_data[i][~plot_d_data[i].isnull().to_numpy()][~plot_o_data[i][~plot_d_data[i].isnull().to_numpy()].isnull().to_numpy()])
        rmse = np.sqrt(np.mean((plot_d_data[i][~plot_d_data[i].isnull().to_numpy()]-plot_o_data[i][~plot_d_data[i].isnull().to_numpy()])**2))
        
        
        ax[i-start][0].set_ylabel(y_labels[i],fontsize=8)
        ax[i-start][1].set_ylabel('Distribution',fontsize=8)
        
        
        ax[i-start][0].scatter(dates,plot_d_data[i],marker='o',color='#999999',s=10)
        ax[i-start][0].scatter(dates,plot_o_data[i],marker='*',color='#377eb8',s=8)
        #ax[i][0].axvline(t_data_mark,color='gray',ls='--')
        #xlimits  = ax[i][0].get_xlim()
        #ylimits = ax[i][0].get_ylim()
        #ax[i][0].fill_between(np.arange(xlimits[0],xlimits[1]), ylimits[1],y2=ylimits[0],\
            #                  where = np.arange(xlimits[0],xlimits[1]) > t_data_mark, facecolor='gray', alpha=.3,label='Test data')
        

        
        ax[i-start][1].hist(plot_d_data[i],bins=20,density=True,color='#999999',edgecolor='black',label='$R_{RS}$ Data')
        ax[i-start][1].hist(plot_o_data[i],bins=20,alpha=0.5,density=True,color='#377eb8',edgecolor='darkblue',label="$R_{RS}$ Model")
        if i == 4:
            rot=20
        else:
            rot=0
        #ax[i-start][0].legend(loc='upper left')
        ax[i-start][0].set_xlabel('year',fontsize=8)
        ax[i-start][1].legend(loc='center right')
        ax[i-start][1].set_xlabel(y_labels[i],fontsize=8)
        ax[i-start][0].set_yticklabels(ax[i-start][0].get_yticklabels(),fontsize=8)
        ax[i-start][1].set_yticklabels(ax[i-start][1].get_yticklabels(),fontsize=8)
        
        my_text1 = "KS Test: {:.3f}, p-value: {:.3E}".format(ks.statistic,ks.pvalue)
        my_text0 = "RMSE: {:.3E}\nCorr: {:.3f}".format(rmse,corr.statistic)
        
        ylimits = ax[i][1].get_ylim()
        ax[i][1].set_ylim(ylimits[0],ylimits[1]+0.2*(ylimits[1]-ylimits[0]))
        xlimits = ax[i][1].get_xlim()
        ax[i][1].set_xlim(xlimits[0],xlimits[1]+corrections[i]*(xlimits[1]-xlimits[0]))
        if i == 3:
            ylimits = ax[i][0].get_ylim()
            ax[i][0].set_ylim(ylimits[0],ylimits[1]+0.1*(ylimits[1]-ylimits[0]))

        
        ax[i-start][0].set_xticklabels(ax[i-start][0].get_xticklabels(), rotation=0, ha='center',fontsize=8)
        ax[i-start][1].set_xticklabels(ax[i-start][1].get_xticklabels(),fontsize=8,rotation=rot)
        ax[i-start][1].text(0.05, 0.95, my_text1, transform=ax[i][1].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)
        ax[i-start][0].text(0.01, 0.95, my_text0, transform=ax[i][0].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)
    plt.show()


def plot_RRS_scatter(second_run_output):
    start,end=0,5
    y_labels = ['$R_{RS,\lambda=412.5}$ ($sr^{-1}$)','$R_{RS,\lambda=442.5}$ ($sr^{-1}$)','$R_{RS,\lambda=490}$ ($sr^{-1}$)','$R_{RS,\lambda=510}$ ($sr^{-1}$)','$R_{RS,\lambda=555}$ ($sr^{-1}$)']
    dates = second_run_output['date']
    plot_d_data = [second_run_output['rrs_d_412'],second_run_output['rrs_d_442'],second_run_output['rrs_d_490'],second_run_output['rrs_d_510'],second_run_output['rrs_d_555']]
    plot_o_data = [second_run_output['RRS_o_412'],second_run_output['RRS_o_442'],second_run_output['RRS_o_490'],second_run_output['RRS_o_510'],second_run_output['RRS_o_555']]

    fig = plt.figure(figsize=(12, 6))
    gs_down = fig.add_gridspec(2,4,width_ratios=[1/6,1/3,1/3,1/6])
    gs_up = fig.add_gridspec(2,3,width_ratios=[1/3,1/3,1/3])
    
    ax1 = fig.add_subplot(gs_up[0,0])
    ax2 = fig.add_subplot(gs_up[0,1])
    ax3 = fig.add_subplot(gs_up[0,2])
    ax4 = fig.add_subplot(gs_down[1,1])
    ax5 = fig.add_subplot(gs_down[1,2])
    
    axes_ = [ax1,ax2,ax3,ax4,ax5]

    for i,ax in enumerate(axes_):
        #if i == 5:
        #    ax.axis("off")
        #    break
        data1 = plot_d_data[i][~plot_d_data[i].isnull().to_numpy()][~plot_o_data[i][~plot_d_data[i].isnull().to_numpy()].isnull().to_numpy()]
        data2 = plot_o_data[i][~plot_d_data[i].isnull().to_numpy()][~plot_o_data[i][~plot_d_data[i].isnull().to_numpy()].isnull().to_numpy()]
        corr = scipy.stats.kendalltau(data1.iloc[:],data2.iloc[:])
        adjust = scipy.stats.linregress(data1.iloc[:],data2.iloc[:])
        rmse = np.sqrt(np.mean(((data1-data2)/data1.std())**2))
        
        
        ax.set_ylabel(y_labels[i]+' Model',fontsize=10)
        ax.set_xlabel(y_labels[i]+' Data',fontsize=10)        
        ax.scatter(data1,data2,marker='o',color='gray',s=8,label='Scatter Plot\nPearson coeffitient: {:.5f}\nRelative RMSE: {:.3f}'.format(adjust.rvalue,rmse))
        #ax.plot(data1,data1*corr.slope + corr.intercept,'--',color='#377eb8',label = 'Line of perfect pearson correlation')
        ax.plot(data1,data1*adjust.slope + adjust.intercept,'--',color='#377eb8',label = 'Line of perfect Pearson correlation')
        ax.legend()
        #ax[i][0].axvline(t_data_mark,color='gray',ls='--')
        #xlimits  = ax[i][0].get_xlim()
        #ylimits = ax[i][0].get_ylim()
        #ax[i][0].fill_between(np.arange(xlimits[0],xlimits[1]), ylimits[1],y2=ylimits[0],\
            #                  where = np.arange(xlimits[0],xlimits[1]) > t_data_mark, facecolor='gray', alpha=.3,label='Test data')
        xticks =   np.linspace( float(ax.get_xticklabels()[0].get_text()) , float(ax.get_xticklabels()[-1].get_text()),6 )
        xticks_labels = ["{:.4f}".format(label) for label in xticks]
        ax.set_xticks(xticks,labels=xticks_labels,fontsize=10,rotation=20)
        #ax.set_xticklabels(ax.get_xticklabels(),fontsize=8,rotation=20)
        
    plt.show()


def plot_kd(second_run_output,error_bar = False):
    dates = second_run_output['date']
    years = np.arange(2005,2013)

    plot_d_data = [second_run_output['kd_d_555'],second_run_output['kd_d_510'],second_run_output['kd_d_490'],second_run_output['kd_d_442'],second_run_output['kd_d_412']]
    plot_o_data = [second_run_output['kd_o_555'],second_run_output['kd_o_510'],second_run_output['kd_o_490'],second_run_output['kd_o_442'],second_run_output['kd_o_412']]
    if error_bar == True:
        plot_o_error = [second_run_output['delta_kd_o_555'],second_run_output['delta_kd_o_510'],second_run_output['delta_kd_o_490'],second_run_output['delta_kd_o_442'],second_run_output['delta_kd_o_412']]
    lambdas = [555,510,490,442.5,412.5]
    fig, ax = plt.subplots(5, 2,width_ratios=[2.5,1],figsize=(18, 9))
    for i in range(5):
        ax[i][0].set_ylabel('$kd_{\lambda='+str(lambdas[4-i])+'}$ ($m^{-1}$)')
        ax[i][1].set_ylabel('Distribution, $\lambda = '+str(lambdas[4-i])+'$')
        years_flags = [False for _ in years]
        ticks_labels = years 
        ticks = []
        year = 0
        for j in range(len(dates)): 
            if (dates.iloc[j].month == 1) and (dates.iloc[j].year == year + 2005) and (years_flags[year] == False):
                ticks.append(np.arange(len(dates))[j])
                years_flags[year] = True
                year+=1

        ks = stats.kstest(plot_d_data[4-i][~(plot_d_data[4-i].isnull()).to_numpy()],plot_o_data[4-i][~(plot_d_data[4-i].isnull()).to_numpy()])
        corr = stats.pearsonr(plot_d_data[4-i][~plot_d_data[4-i].isnull().to_numpy()],\
                          plot_o_data[4-i][~plot_d_data[4-i].isnull().to_numpy()])
        rmse = np.sqrt(np.mean((plot_d_data[4-i][~plot_d_data[4-i].isnull().to_numpy()]-plot_o_data[4-i][~plot_d_data[4-i].isnull().to_numpy()])**2))

    
        ax[i][0].scatter(dates,plot_d_data[4-i],marker = 'o',label='Filtered data',color='black',s=10)
        if error_bar == False:
            ax[i][0].scatter(dates,plot_o_data[4-i],marker ='*',label='Daily minimization',color='#377eb8',alpha=0.4,s=10) #377eb8 seagreen
        else:
            data_plot = pd.DataFrame({'error_up':plot_o_error[4-i].abs(),'error_down':plot_o_error[4-i].abs(),'zero':np.zeros(len(plot_o_error[4-i])),'data':plot_o_data[4-i]})
            data_plot['error_down'] = data_plot[['error_down','data']].min(axis=1)
            ax[i][0].errorbar(dates, plot_o_data[4-i], yerr=[data_plot['error_down'],plot_o_error[4-i].abs()], capsize=2, fmt="o", c='#377eb8',ms=2,alpha=0.4)


    
        #ax[i][0].set_xticks(ticks)
        #ax[i][0].set_xticklabels(ticks_labels, rotation=0, ha='center')
        ax[i][1].hist(plot_d_data[4-i][~plot_d_data[4-i].isnull().to_numpy()][~plot_o_data[4-i][~plot_d_data[4-i].isnull().to_numpy()].isnull().to_numpy()],bins=20,label='Filtered data',density=True,color='black',edgecolor='gray')
        ax[i][1].hist(plot_o_data[4-i][~plot_d_data[4-i].isnull().to_numpy()][~plot_o_data[4-i][~plot_d_data[4-i].isnull().to_numpy()].isnull().to_numpy()],bins=20,label='Daily minimization',alpha=0.5,density=True,color='#377eb8',edgecolor='blue')
        ax[i][0].legend(loc='upper left')
        ax[i][0].set_xlabel('year')
        ax[i][1].legend(loc='upper right')
        ax[i][1].set_xlabel('$kd (m^{-1}$)')
        my_text = "KS Test: {:.3f}, p-value: {:.3E}\nRMSE: {:.3E}\nCorr: {:.3f}".format(ks.statistic,ks.pvalue,rmse,corr.statistic)
        ax[i][1].text(0.4, 0.5, my_text, transform=ax[i][1].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)
        fig.tight_layout()
    plt.show()

    
def plot_bbp(second_run_output,error_bar = False):
    dates = second_run_output['date']
    years = np.arange(2005,2013)
    plot_d_data = [second_run_output['bbp_d_555'],second_run_output['bbp_d_490'],second_run_output['bbp_d_442'],second_run_output['chla_d']]
    plot_o_data = [second_run_output['bbp_o_555'],second_run_output['bbp_o_490'],second_run_output['bbp_o_442'],second_run_output['chla_o']]
    if error_bar == True:
        plot_o_error = [second_run_output['delta_bbp_o_555'],second_run_output['delta_bbp_o_490'],second_run_output['delta_bbp_o_442'],second_run_output['delta_chla_o']]
    lambdas = [555,490,442.5]
    fig, ax = plt.subplots(4, 2,width_ratios = [2.5,1],figsize=(18, 9))
    for i in range(4):
        if i < 3:
            ax[i][0].set_ylabel('$b_{b,p,\lambda='+str(lambdas[2-i])+'}(m^{-1})$')
            ax[i][1].set_ylabel('Distribution, $\lambda = '+str(lambdas[i])+'$')
            ax[i][1].set_xlabel('$ b_{b,p}(m^{-1})$')
        else:
            ax[i][0].set_ylabel('$Chlorophyll-a (mg/m^{3})$')
            ax[i][1].set_ylabel('Distribution, Chlorophyll-a')
            ax[i][1].set_xlabel('$Chlorophyll-a (mg/m^{3})$')

        years_flags = [False for _ in years]
        ticks_labels = years 
        ticks = []
        year = 0

        for j in range(len(dates)): 
            if (dates.iloc[j].month == 1) and  (dates.iloc[j].year == year + 2005) and (years_flags[year] == False):
                ticks.append(np.arange(len(dates))[j])
                years_flags[year] = True
                year+=1

        ks = stats.kstest(plot_d_data[2-i][~(plot_d_data[2-i].isnull()).to_numpy()],plot_o_data[2-i][~(plot_d_data[2-i].isnull()).to_numpy()])
        corr = stats.pearsonr(plot_d_data[2-i][~plot_d_data[2-i].isnull().to_numpy()],\
                          plot_o_data[2-i][~plot_d_data[2-i].isnull().to_numpy()])
        rmse = np.sqrt(np.mean((plot_d_data[2-i][~plot_d_data[2-i].isnull().to_numpy()]-plot_o_data[2-i][~plot_d_data[2-i].isnull().to_numpy()])**2))
        
        ax[i][0].scatter(dates,plot_d_data[2-i],marker='o',label='Filtered data',color='black',s=10)
        if error_bar == False:
            ax[i][0].scatter(dates,plot_o_data[2-i],marker='x',label='Daily minimization',color='#377eb8',s=10,alpha=0.4)
        else:
            data_plot = pd.DataFrame({'error_up':plot_o_error[2-i].abs(),'error_down':plot_o_error[2-i].abs(),'zero':np.zeros(len(plot_o_error[2-i])),'data':plot_o_data[2-i]})
            data_plot['error_down'] = data_plot[['error_down','data']].min(axis=1)
            ax[i][0].errorbar(dates, plot_o_data[2-i], yerr=[data_plot['error_down'],plot_o_error[2-i]], capsize=2, fmt="o", c='#377eb8',ms=2,alpha=0.4)
        #ax[i][0].set_xticks(ticks)
        #ax[i][0].set_xticklabels(ticks_labels, rotation=0, ha='center')
    
        ax[i][1].hist(plot_d_data[2-i][~plot_d_data[2-i].isnull().to_numpy()][~plot_o_data[2-i][~plot_d_data[2-i].isnull().to_numpy()].isnull().to_numpy()],bins=20,label='Filtered data',density=True,color='black',edgecolor='gray')
        ax[i][1].hist(plot_o_data[2-i][~plot_d_data[2-i].isnull().to_numpy()][~plot_o_data[2-i][~plot_d_data[2-i].isnull().to_numpy()].isnull().to_numpy()],bins=20,label='Daily Minimization',alpha=0.5,density=True,color='#377eb8',edgecolor='blue') #seagreen, darkgreen
        ax[i][0].legend(loc='upper left')
        ax[i][0].set_xlabel('year')
        ax[i][1].legend(loc='upper right')
        my_text = "KS Test: {:.3f}, p-value: {:.3E}\nRMSE: {:.3E}\nCorr: {:.3f}".format(ks.statistic,ks.pvalue,rmse,corr.statistic)
        ax[i][1].text(0.4, 0.5, my_text, transform=ax[i][1].transAxes, fontsize=8, verticalalignment='top')#, bbox=props)

        fig.tight_layout()
    plt.show()
    


init_time = time.time()
initial_conditions_path = '/Users/carlos/Documents/surface_data_analisis/results_first_run'
data_path = '/Users/carlos/Documents/surface_data_analisis/SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS'

data = read_train_data(initial_conditions_path,data_path,first_run=True)
second_run_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/results_bayes_optimized'  
second_run_output = read_second_run(second_run_path,include_uncertainty=True,optimized = True)
second_run_output = add_messure_data(second_run_output,data,include_first_run=False)
plot_kd(second_run_output,error_bar=True)
plot_bbp(second_run_output,error_bar=True)
plot_RRS_scatter(second_run_output)
#plot_RRS_data(second_run_output)
