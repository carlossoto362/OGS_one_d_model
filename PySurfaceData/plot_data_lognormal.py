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

from ModelOnePointFive import *

import warnings

MODEL_HOME = '/Users/carlos/Documents/OGS_one_d_model'

def data_dataframe(data_path,which='all'):
    data = customTensorData(data_path=data_path,which=which,per_day = False,randomice=False)
    
    dataframe = pd.DataFrame(columns = data.x_column_names + data.y_column_names + data.init_column_names)
    #print(data.init_column_names,data.init.shape)
    dataframe[data.x_column_names] = data.x_data
    dataframe[data.y_column_names] = data.y_data
    dataframe[data.init_column_names] = data.init
    dataframe['date'] = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in data.dates]
    dataframe.sort_values(by='date',inplace=True)
    return dataframe
    


def read_second_run(second_run_path,include_uncertainty=False,abr='output',name_index = None, ignore_name = None):
    if include_uncertainty == False:
        second_run_output = pd.DataFrame(columns=['RRS_'+abr+'_412','RRS_'+abr+'_442','RRS_'+abr+'_490','RRS_'+abr+'_510','RRS_'+abr+'_555',\
                                                  'chla_'+abr,'NAP_'+abr,'CDOM_'+abr,\
                                                  'kd_'+abr+'_412','kd_'+abr+'_442','kd_'+abr+'_490','kd_'+abr+'_510',\
                                                  'kd_'+abr+'_555','bbp_'+abr+'_442','bbp_'+abr+'_490','bbp_'+abr+'_555'])
        len_kd = 5
        len_bbp = 3
        len_chla = 3
    else:

        second_run_output = pd.DataFrame(columns=['RRS_'+abr+'_412','RRS_'+abr+'_442','RRS_'+abr+'_490','RRS_'+abr+'_510','RRS_'+abr+'_555',\
                                                  'chla_'+abr, 'delta_chla_'+abr,'NAP_'+abr,'delta_NAP_'+abr,'CDOM_'+abr,'delta_CDOM_'+abr,\
                                                  'kd_'+abr+'_412','delta_kd_'+abr+'_412','kd_'+abr+'_442','delta_kd_'+abr+'_442','kd_'+abr+'_490','delta_kd_'+abr+'_490','kd_'+abr+'_510','delta_kd_'+abr+'_510',\
                                                  'kd_'+abr+'_555','delta_kd_'+abr+'_555','bbp_'+abr+'_442','delta_bbp_'+abr+'_442','bbp_'+abr+'_490','delta_bbp_'+abr+'_490','bbp_'+abr+'_555','delta_bbp_'+abr+'_555'])
        len_kd = 10
        len_bbp = 6
        len_chla = 6

    if name_index == None:
        RRS_name = 'RRS_hat.npy'
        X_name = 'X_hat.npy'
        kd_name = 'kd_hat.npy'
        bbp_name = 'bbp_hat.npy'
    else:
        RRS_name = 'RRS_hat'+'_'+name_index+'.npy'
        X_name = 'X_hat'+'_'+name_index+'.npy'
        kd_name = 'kd_hat'+'_'+name_index+'.npy'
        bbp_name = 'bbp_hat'+'_'+name_index+'.npy'

    second_run_output[second_run_output.columns[:5]] = np.load(second_run_path + '/'+RRS_name)
    second_run_output[second_run_output.columns[5:5+len_chla]] = np.load(second_run_path + '/'+X_name)
    second_run_output[second_run_output.columns[5+len_chla:5+len_chla + len_kd]] = np.load(second_run_path + '/'+kd_name)
    second_run_output[second_run_output.columns[5+len_chla + len_kd:]] = np.load(second_run_path + '/'+bbp_name)
    dates = np.load(second_run_path + '/dates.npy')
    second_run_output['date'] = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in dates]
    second_run_output.sort_values(by='date',inplace=True)

    return second_run_output
         



def plot_parallel(data,columns,names,labels,statistics = True,histogram=True,figsize=(30,17),date_init = None,\
                  date_end = None, shadow_error = False,num_cols = 2,figname = 'fig.png',fontsize=15,colors = 1,save=True,indexes=[],ylim=[],legend_fontsize=12,s=2):
    if colors == 1:
        colors_palet = ['#377eb8','blue']
    elif colors == 2:
        colors_palet = ['#FFC107','#D81B60']
    elif colors == 3:
        colors_palet = ['#15CAAB','#247539']
    
    if date_init != None:
        data_use = data[data['date'] >= date_init]
    else:
        data_use = data
    if date_end != None:
        data_use = data_use[data_use['date'] <= date_end]
        
    dates = data_use['date']
    num_plots = math.ceil(len(columns)/num_cols)

    
    if histogram == True:
        fig,axs = plt.subplots(num_plots,num_cols*2,width_ratios = [2.5/num_cols,1/num_cols]*num_cols,figsize = figsize,tight_layout=True)
    else:
        fig,axs = plt.subplots(num_plots,num_cols,figsize = figsize,tight_layout=True)

    k = 0
    for j in range(num_cols):
        if j == num_cols - 1:
            end_enum = (len(columns) - num_plots*(num_cols - 1) ) + (num_plots)*j

            if histogram == True:
                for ax in axs[(len(columns) - num_plots*(num_cols - 1) ):]:
                    ax[-2].axis('off')
                    ax[-1].axis('off')
            elif len(columns)>1:
                for ax in axs.flatten()[len(columns):]:
                    ax.axis('off')
        else:
            end_enum = (num_plots)*(j+1)
        for i,column in enumerate(columns[j*(num_plots):end_enum]):

            
            if statistics == True:
                if pd.isnull(data_use[column[0]]).all():
                   statistic_text = ''
                else:
                    if (column[0].split('_')[0] == 'chla') or (column[0].split('_')[0] == 'NAP') or (column[0].split('_')[0] == 'CDOM'):
                        
                        data_statistics_1 = np.exp((data_use[column[0]][~data_use[column[0]].isnull()]))
                        data_statistics_2 = np.exp((data_use[column[1]][~data_use[column[0]].isnull()]))
                    else:
                        data_statistics_1 = (data_use[column[0]][~data_use[column[0]].isnull()])
                        data_statistics_2 = (data_use[column[1]][~data_use[column[0]].isnull()])

                    ks = stats.kstest(data_statistics_1,data_statistics_2)
                    corr = stats.pearsonr(data_statistics_1,data_statistics_2)
                    rmse = np.sqrt(np.mean((data_statistics_1-data_statistics_2)**2))
                    bias = np.mean((data_statistics_1-data_statistics_2))
                    statistic_text = "KS Test: {:.3f}, p-value: {:.3E}\nRMSE: {:.3E}\nCorrelation: {:.3f}\nBias: {:.3E}".format(ks.statistic,ks.pvalue,rmse,corr.statistic,bias)

            if len(columns) >1:
                if histogram == True:
                    ax = axs[i,2*j]
                elif num_cols == 1:
                    ax = axs[i]
                    
                else:
                    ax = axs[i,j]
            else:
                if histogram == True:
                    ax = axs[0]
                else:
                    ax = axs
                

            if len(column) >= 1:
                if pd.isnull(data_use[column[0]]).all():
                    pass
                else:
                    if (column[0].split('_')[0] == 'chla') or (column[0].split('_')[0] == 'NAP') or (column[0].split('_')[0] == 'CDOM'):
                        print(column[0])
                        ax.scatter(dates,np.exp(data_use[column[0]]),marker = 'o',label=labels[k][0],color='black',s=s,zorder = 10,alpha=0.7)
                    else:
                        ax.scatter(dates,data_use[column[0]],marker = 'o',label=labels[k][0],color='black',s=s,zorder = 10,alpha=0.7)
                ax.set_xlabel('date',fontsize=fontsize)
                ax.set_ylabel(names[k],fontsize=fontsize)
                if (statistics == True) and (histogram == False):
                    ax.text(0.99, 0.95, statistic_text, transform=ax.transAxes, fontsize=fontsize*0.7, verticalalignment='top',horizontalalignment = 'right',bbox = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='white'),zorder=20)
            
            if len(column) == 2:
                if (column[0].split('_')[0] == 'chla') or (column[0].split('_')[0] == 'NAP') or (column[0].split('_')[0] == 'CDOM'):
                    ax.scatter(dates,np.exp(data_use[column[1]]),marker ='*',label=labels[k][1],color=colors_palet[0],alpha=0.6,s=s,zorder = 5)
                else:
                    ax.scatter(dates,data_use[column[1]],marker ='*',label=labels[k][1],color=colors_palet[0],alpha=0.6,s=s,zorder = 5)
            if len(column) >= 3:
                if (column[0].split('_')[0] == 'chla') or (column[0].split('_')[0] == 'NAP') or (column[0].split('_')[0] == 'CDOM'):
                    median = np.exp(data_use[column[1]])
                    intervals = scipy.stats.lognorm.interval(0.68,data_use[column[2]],scale = np.exp(data_use[column[1]]))
                    data_plot = pd.DataFrame({'error_up': intervals[1] - median,\
                                              'error_down':median - intervals[0],'data':median})
                else:
                    data_plot = pd.DataFrame({'error_up':data_use[column[2]].abs(),'error_down':data_use[column[2]].abs(),'data':data_use[column[1]]})
                    data_plot['error_down'] = data_plot[['error_down','data']].min(axis=1)
                    
                if shadow_error == False:
                    ax.errorbar(dates, data_plot['data'], yerr=[data_plot['error_down'],data_plot['error_up']], capsize=2, fmt="o", c=colors_palet[1],ecolor=colors_palet[0],ms=2,alpha=0.6,zorder=5,label = labels[k][1])
                else:
                    if (column[0].split('_')[0] == 'chla') or (column[0].split('_')[0] == 'NAP') or (column[0].split('_')[0] == 'CDOM'):
                        ax.scatter(dates,np.exp(data_use[column[1]]),marker ='x',label=labels[k][1],color=colors_palet[1],alpha=0.6,s=5*s,linewidths=0.5,zorder = 5)
                    else:
                        ax.scatter(dates,data_use[column[1]],marker ='x',label=labels[k][1],color=colors_palet[1],alpha=0.6,s=5*s,linewidths=0.5,zorder = 5)
                    ax.fill_between(dates, data_plot['data']-data_plot['error_down'], data_plot['data']+data_plot['error_up'],color=colors_palet[0],alpha=0.6,zorder = 1)
            if len(column) == 4:
                if (column[0].split('_')[0] == 'chla') or (column[0].split('_')[0] == 'NAP') or (column[0].split('_')[0] == 'CDOM'):
                    ax.scatter(dates,np.exp(data_use[column[3]]),marker='+',label=labels[k][-1],color = '#D81B60',alpha=1,zorder=50,s=5*s,linewidths=0.5)
                else:
                    ax.scatter(dates,data_use[column[3]],marker='+',label=labels[k][-1],color = '#D81B60',alpha=1,zorder=50,s=5*s,linewidths=0.5)

            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
        
            if histogram == True:
                ax = axs[i,2*j + 1]
                if len(column) >= 1:
                    if (column[0].split('_')[0] == 'chla') or (column[0].split('_')[0] == 'NAP') or (column[0].split('_')[0] == 'CDOM'):
                        data_plot = np.exp(data_use[column[0]][~data_use[column[0]].isnull()])
                    else:
                        data_plot = data_use[column[0]][~data_use[column[0]].isnull()]
                    ax.hist(data_plot,bins=20,label=labels[k][2],density=True,color='black',edgecolor='gray')
                    ax.set_xlabel(names[k],fontsize=fontsize)
                    #ax.set_xlim(0,0.3)
                    if (statistics == True):
                        ax.text(0.99, 0.7, statistic_text, transform=ax.transAxes, fontsize=fontsize*0.7, verticalalignment='top',horizontalalignment = 'right',bbox = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='white'),zorder = 20)
                    if len(column) >= 2:
                        if (column[0].split('_') == 'chla') or (column[0].split('_') == 'NAP') or (column[0].split('_') == 'CDOM'):
                            data_plot = np.exp(data_use[column[1]][~data_use[column[0]].isnull()])
                        else:
                            data_plot = data_use[column[1]][~data_use[column[0]].isnull()]
                        ax.hist(data_plot,bins=20,label=labels[k][3],alpha=0.5,density=True,color=colors_palet[0],edgecolor=colors_palet[1])
                        
                ax.legend(loc='upper right',fontsize=legend_fontsize)
                ax.tick_params(axis='x', labelsize=fontsize)
                ax.tick_params(axis='y', labelsize=fontsize)
            if column[0].split('_')[0] == 'kd':
                ax.set_ylim(1e-2,0.25)
                #print(column[0])
                #print(data_use[column[0]].min())
                    
                #ax.set_yscale('log')
            if column[0].split('_')[0] == 'bbp':
                ax.set_ylim(1e-4,0.005)
                #ax.set_yscale('log')
            ax.legend(fontsize=legend_fontsize)
            if len(indexes)>0:
                ax.text(-0.03,1.13,indexes[k],transform = ax.transAxes,fontsize=fontsize)
                
            k+=1

            if len(ylim)==0:
                pass
            else:
                ax.set_ylim(ylim[0],ylim[1])
        
    fig.tight_layout()
    if save == True:
        
        plt.savefig(figname)
    else:
        plt.show()
    
def plot_kd(input_data_path = MODEL_HOME + '/results_bayes_lognormal_logparam',ylim=[],\
                figname = MODEL_HOME + '/plot_data/kd_lognormal_data.pdf',save=True,date_init = datetime(year=2005,month=1,day=1),\
                third_data_path = MODEL_HOME + '/results_VAE_VAEparam',statistics=False, num_cols = 1,figsize=(30,17),labels_names=['In situ data','Bayesian Alternate Minimization']):

    data = data_dataframe( MODEL_HOME + '/npy_data')

    data['NAP'] = np.nan
    data['CDOM'] = np.nan
    second_run = read_second_run(input_data_path,include_uncertainty=True,abr='output')
    data = second_run.merge(data,how='right',on='date')
    second_run = read_second_run(third_data_path,include_uncertainty=True,abr='outputVAE')
    data = second_run.merge(data,how='right',on='date')
    data.sort_values(by='date',inplace=True)
    del second_run

    
    lambdas_names = ['412','442','490','510','555']
    lambdas_values = ['412.5','442.5','490','510','555']

    columns = []
    names = []
    labels = []
    indexes = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']

    for i,lam in enumerate(lambdas_names):
        columns.append(('kd_'+ lam,'kd_output_' + lam,'delta_kd_output_'+lam,'kd_outputVAE_'+lam))
        names.append('$kd_{'+lambdas_values[i]+'} [\\text{m}^{-1}$]')
        labels.append((*labels_names,*labels_names,'Generative Neural Network Output'))


    plot_parallel(data,columns,names,labels,statistics = statistics,histogram=False,date_init = date_init,shadow_error = True,num_cols=num_cols,\
                  figname = figname,fontsize=25,colors = 1,save=save,figsize = figsize,indexes = indexes,ylim=ylim)

def plot_bbp(input_data_path = MODEL_HOME + '/results_bayes_lognormal_logparam',ylim=[],\
                figname = MODEL_HOME + '/plot_data/bbp_lognormal_data.pdf',save=True,date_init = datetime(year=2005,month=1,day=1),\
                third_data_path = MODEL_HOME + '/results_VAE_VAEparam',statistics=False, num_cols = 2,figsize=(30,17),labels_names=['In situ data','Bayesian Alternate Minimization']):

    data = data_dataframe(MODEL_HOME + '/npy_data')

    data['NAP'] = np.nan
    data['CDOM'] = np.nan
    second_run = read_second_run(input_data_path,include_uncertainty=True,abr='output')
    data = second_run.merge(data,how='right',on='date')
    second_run = read_second_run(third_data_path,include_uncertainty=True,abr='outputVAE')
    data = second_run.merge(data,how='right',on='date')
    data.sort_values(by='date',inplace=True)
    del second_run

    
    lambdas_names = ['412','442','490','510','555']
    lambdas_values = ['412.5','442.5','490','510','555']

    columns = []
    names = []
    labels = []
    indexes = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']

    for i,lam in enumerate(lambdas_names):
        if (i == 1) or (i == 2) or (i ==4):
            columns.append(('bbp_'+ lam,'bbp_output_' + lam,'delta_bbp_output_'+lam,'bbp_outputVAE_' + lam))
            names.append('$b_{b,p,'+lambdas_values[i]+'} [m^{-1}$]')
            labels.append((*labels_names,*labels_names,'Generative Neural Network Output'))

    plot_parallel(data,columns,names,labels,statistics = statistics,histogram=False,date_init = date_init,shadow_error = True,num_cols=num_cols,\
                  figname = figname,fontsize=25,colors = 1,save=save,figsize = figsize,indexes = indexes,ylim=ylim)

def plot_chla(input_data_path = MODEL_HOME + '/results_bayes_lognormal_logparam',ylim=[],\
                figname = MODEL_HOME + '/plot_data/chla_lognormal_data.pdf',save=True,date_init = datetime(year=2005,month=1,day=1),\
                third_data_path = MODEL_HOME + '/results_VAE_VAEparam',statistics=False, num_cols = 2,figsize=(30,17),labels_names=['In situ data','Bayesian Alternate Minimization']):

    data = data_dataframe( MODEL_HOME + '/npy_data')

    data['NAP'] = np.nan
    data['CDOM'] = np.nan
    second_run = read_second_run(input_data_path,include_uncertainty=True,abr='output')
    data = second_run.merge(data,how='right',on='date')
    second_run = read_second_run(third_data_path,include_uncertainty=True,abr='outputVAE')
    data = second_run.merge(data,how='right',on='date')
    data.sort_values(by='date',inplace=True)
    del second_run

    
    lambdas_names = ['412','442','490','510','555']
    lambdas_values = ['412.5','442.5','490','510','555']

    columns = []
    names = []
    labels = []
    indexes = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']


    columns.append(('chla','chla_output','delta_chla_output','chla_outputVAE'))
    names.append('$\\text{Chl-a } [\\text{mg}\\text{m}^{-3}]$')
    labels.append((*labels_names,*labels_names,'Generative Neural Network Output'))

    columns.append(('NAP','NAP_output','delta_NAP_output','NAP_outputVAE'))
    names.append('$\\text{NAP } [\\text{mg} \\text{m}^{-3}]$')
    labels.append((*labels_names,*labels_names,'Generative Neural Network Output'))

    columns.append(('CDOM','CDOM_output','delta_CDOM_output','CDOM_outputVAE'))
    names.append('$\\text{CDOM } [\\text{mg}\\text{m}^{-3}]$')
    labels.append((*labels_names,*labels_names,'Generative Neural Network Output'))

    plot_parallel(data,columns,names,labels,statistics = statistics,histogram=False,date_init = date_init,shadow_error = True,num_cols=num_cols,\
                  figname = figname,fontsize=25,colors = 1,save=save,figsize = figsize,indexes = indexes,ylim=ylim)

def comparison_alphas():
    data = data_dataframe(MODEL_HOME + '/npy_data',which = 'all')
    data['chla'] = np.exp(data['chla'])
    data['NAP'] = np.nan
    data['CDOM'] = np.nan

    second_runs_errors = pd.DataFrame(columns = ['alpha','epsilon_rrs','error_output_mean','error_mean','epsilon_error'])
    alphas = []
    k=0
    for file_ in os.listdir(MODEL_HOME + '/results_bayes_lognormal_logparam/alphas'):
        if file_ == 'dates.npy':
            continue
        
        name_ = file_.split('_')
        alpha = '.'.join(name_[2].split('.')[:2])
        if (alpha in alphas):
            pass
        else:
            alphas.append(alpha)
            second_run_i = read_second_run(MODEL_HOME + '/results_bayes_lognormal_logparam/alphas',include_uncertainty=True,abr='output',name_index = alpha )

            second_run_i['delta_chla_output'] = np.sqrt(scipy.stats.lognorm.var(second_run_i['delta_chla_output'],scale=np.exp(second_run_i['chla_output'])))
            second_run_i['chla_output'] = scipy.stats.lognorm.median(second_run_i['delta_chla_output'],scale=np.exp(second_run_i['chla_output']))
            epsilon_rrs = data[data.columns[:5]].to_numpy() - second_run_i[second_run_i.columns[:5]].to_numpy()
            epsilon_rrs = np.mean(np.sqrt(np.mean(epsilon_rrs**2,axis=1)))
            error_output_mean = np.nanmean(second_run_i[second_run_i.columns[6]].to_numpy())

            error_mean = np.abs(data[data.columns[22]].to_numpy() - second_run_i[second_run_i.columns[5]].to_numpy())
            error_mean = np.mean(np.sqrt(np.nanmean(error_mean**2)))
            epsilon_error = np.abs(error_mean - error_output_mean)

            second_runs_errors.loc[len(second_runs_errors)] = [float(alpha),epsilon_rrs,error_output_mean,error_mean,epsilon_error]
            k+=1

    second_runs_errors.sort_values(by='alpha',inplace = True)
    second_runs_errors = second_runs_errors[second_runs_errors['alpha']<6]

    normilized_epsilon_rrs = (((second_runs_errors['epsilon_rrs'] -  second_runs_errors['epsilon_rrs'].mean())/second_runs_errors['epsilon_rrs'].std()) )
    normilized_epsilon_error = (((second_runs_errors['epsilon_error'] -  second_runs_errors['epsilon_error'].mean())/second_runs_errors['epsilon_error'].std()))
    normilized_error_chla = (((second_runs_errors['error_mean'] -  second_runs_errors['error_mean'].mean())/second_runs_errors['error_mean'].std()))

    normilized_factors = second_runs_errors[['epsilon_rrs','epsilon_error','error_mean']].max(axis=1).max()

    normilized_epsilon_rrs = (second_runs_errors['epsilon_rrs'] - second_runs_errors['epsilon_rrs'].min())/(second_runs_errors['epsilon_rrs'] - second_runs_errors['epsilon_rrs'].min()).max()
    normilized_epsilon_error = (second_runs_errors['epsilon_error'] - second_runs_errors['epsilon_error'].min())/(second_runs_errors['epsilon_error'] - second_runs_errors['epsilon_error'].min()).max()
    normilized_error_chla = (second_runs_errors['error_mean'] - second_runs_errors['error_mean'].min())/(second_runs_errors['error_mean'] - second_runs_errors['error_mean'].min()).max()


    Loss_function =  normilized_epsilon_rrs  + normilized_epsilon_error#/(normilized_epsilon_rrs +  normilized_epsilon_error).max()
    fig,ax = plt.subplots(3,1)
    colors = plt.cm.viridis(np.linspace(0,1,17))

    f = scipy.interpolate.interp1d(second_runs_errors['alpha'],second_runs_errors['epsilon_rrs'],kind='cubic')
    xnew = np.arange(second_runs_errors['alpha'].min(), second_runs_errors['alpha'].max(), 0.01)
    ynew = f(xnew)
    ax[0].plot(xnew,ynew,'--',color='gray',alpha=0.4)

    

    f = scipy.interpolate.interp1d(second_runs_errors['alpha'], second_runs_errors['epsilon_error'],kind='cubic')
    xnew = np.arange(second_runs_errors['alpha'].min(), second_runs_errors['alpha'].max(), 0.01)
    ynew = f(xnew)
    ax[1].plot(xnew,ynew,'--',color = colors[1],alpha=0.4)


    f = scipy.interpolate.interp1d(second_runs_errors['alpha'], Loss_function,kind='cubic')
    xnew = np.arange(second_runs_errors['alpha'].min(), second_runs_errors['alpha'].max(), 0.01)
    ynew = f(xnew)
    ax[2].plot(xnew,ynew,'--',color = colors[5], alpha= 0.4)


    s = 15
    ax[0].scatter(second_runs_errors['alpha'],second_runs_errors['epsilon_rrs'],label='$\epsilon_{R_{RS}} =  RMSD(R_{RS}^{OBS},R_{RS}^{MOD})$',c='black',s=s)
    
    ax[1].scatter(second_runs_errors['alpha'],second_runs_errors['epsilon_error'],label='$\epsilon_{\delta_{chla}} =  MEAN(|RMSD(chla^{OBS} , chla^{MOD}) - MEAN(\delta_{chla})|)$',c = colors[1],marker='x',s=s)
    ax[2].scatter(second_runs_errors['alpha'],Loss_function,label='$\mathbf{L} =\overline{ \epsilon_{R_{RS}} }  + \overline{  \epsilon_{\delta_{chla}}}$',c = colors[5],marker='^',s=s)
    
    ax[0].axvline(4.87,linestyle='--',color='red')
    ax[1].axvline(4.87,linestyle='--',color='red')
    ax[2].axvline(4.87,linestyle='--',color='red')
    ax[2].text(4.88,1,'$\\text{argmin}_{\\alpha}(\mathbf{L}$)',color=colors[5])

    ax[0].text(-0.1,1.05,'(a)',transform = ax[0].transAxes)
    ax[1].text(-0.1,1.05,'(b)',transform = ax[1].transAxes)
    ax[2].text(-0.1,1.05,'(c)',transform = ax[2].transAxes)
    ax[0].set_ylabel('$\epsilon_{R_{RS}}$ [$sr^{-1}$]')
    ax[1].set_ylabel('$\epsilon_{\delta_{chla}}$ [$mg m^{-3}$]')
    ax[2].set_ylabel('$L^{\\alpha}$')
    

    for axis in ax:
        #axis.legend(fontsize="10.5",bbox_to_anchor=(0.1,0.65),loc='lower left',shadow=False)
        axis.legend(fontsize='10.5')

    ax[2].set_xlabel('A-priori covariance for chl-a, NAP and CDOM ($\\alpha$)')
    #ax[1].set_ylabel('Normilized errors')
    for axis in ax[:2]:
        axis.xaxis.set_visible(False)
    fig.tight_layout()
    plt.show()
            


                
def plot_scaterplot(test_indexes , vae_path = MODEL_HOME + '/results_VAE_VAEparam' ):
    data = data_dataframe(MODEL_HOME + '/npy_data')

    data['NAP'] = np.nan
    data['CDOM'] = np.nan
    second_run = read_second_run(MODEL_HOME + '/results_bayes_lognormal_logparam',include_uncertainty=True,abr='log_output')
    data = second_run.merge(data,how='right',on='date')
    second_run = read_second_run(MODEL_HOME + '/results_bayes_lognormal_VAEparam',include_uncertainty=True,abr='logNN_output')
    data = second_run.merge(data,how='right',on='date')
    second_run = read_second_run(MODEL_HOME + '/results_bayes_lognormal_unperturbed',include_uncertainty=True,abr='un_output')
    data = second_run.merge(data,how='right',on='date')
    second_run = read_second_run(MODEL_HOME + '/results_NN_NNparam',include_uncertainty=False,abr='NN_output')
    data = second_run.merge(data,how='right',on='date')
    second_run = read_second_run(vae_path,include_uncertainty=True,abr='VAE_output')
    data = second_run.merge(data,how='right',on='date')
    data.sort_values(by='date',inplace=True)
    del second_run



    indexes = ((data['date']>datetime(year=2012,month=5,day=1) ) & (data['date']<datetime(year=2012,month=9,day=1) ) )
    print(data['bbp_442'][indexes].mean())
    print(data['bbp_490'][indexes].mean())
    print(data['bbp_555'][indexes].mean())

    indexes = data['date']>=datetime(year=2012,month=1,day=1)
    plt.plot(data['date'][indexes],data['chla'][indexes],label='data')
    plt.plot(data['date'][indexes],data['chla_VAE_output'][indexes],label='model vae')
    plt.legend()
    plt.show()
    





    
    abrs = ['log_output','logNN_output','un_output','NN_output','VAE_output']

    for abr in abrs :
        total_error = 0
        for var in ['chla_'+abr,*['kd_' + abr + '_' + lamb for lamb in ['555','510','490','442','412']],*['bbp_' + abr + '_' + lamb for lamb in ['555','490','442']]]:
            if var.split('_')[0] == 'chla':
                total_error += np.sqrt(np.nanmean((data[var.split('_')[0]].iloc[test_indexes] - data[var].iloc[test_indexes] )**2))
            else:

                total_error += np.sqrt(np.nanmean((data[var.split('_')[0]  + '_' + var.split('_')[-1]].iloc[test_indexes]  - data[var].iloc[test_indexes] )**2))
        print(abr,total_error)

    data['month'] = [data['date'].iloc[i].month for i in range(len(data['date']))]




    #plot_scatter_plot
    fig,axs = plt.subplots(2,2,tight_layout=True)

    def plot_one_line(ax,data1,data2,color='black',marker='o', label= '',test='False'):

        data1_ = data1.iloc[test_indexes]
        data2_ = data2.iloc[test_indexes]
        data1_ = data1[~data2.isnull()][~data1[~data2.isnull()].isnull()]
        data2_ = data2[~data2.isnull()][~data1[~data2.isnull()].isnull()]
        f = scipy.stats.linregress(data1_,data2_)
        if test==True:
            print('slope',f.slope)
            print('{:.5f}'.format(stats.pearsonr(data1_,data2_).statistic))
            print(data1_)

        data1_ = [np.nanmean(data1[data['month'] == i]) for i in range(1,13)]
        data2_ = [np.nanmean(data2[data['month'] == i]) for i in range(1,13)]
        x = np.linspace(np.min(data1_),np.max(data1_),20)

        ax.scatter(data1_,data2_,alpha=0.8,marker = marker,color = color, label = label)
        ax.plot(x,f.slope * x + f.intercept,'--',alpha=0.8,color = color)

        return data1_

    
    plot_one_line(axs[0,0],np.exp(data['chla']),np.exp(data['chla_VAE_output']),color='#CE2864',marker='x',label = 'SGVB')
    plot_one_line(axs[0,0],np.exp(data['chla']),np.exp(data['chla_log_output']),color='#1E88E5',marker='o',label = 'MAP AM param',test=True)
    #plot_one_line(axs[0,0],np.exp(data['chla']),np.exp(data['chla_NN_output']),color='green',marker='x',label = 'Neural Network')
    plot_one_line(axs[0,0],np.exp(data['chla']),np.exp(data['chla_logNN_output']),color='#674E03',marker='*',label = 'MAP SGVB param',test=True)
    data_1 = plot_one_line(axs[0,0],np.exp(data['chla']),np.exp(data['chla_un_output']),color='#004D40',marker='v',label = 'MAP Unperturbed param')
    axs[0,0].plot(np.linspace(np.min(data_1),np.max(data_1),20),np.linspace(np.min(data_1),np.max(data_1),20),alpha=0.3,lw=3,label='Perfect Linear Correlation',color ='gray',zorder=50)
    
    axs[0,0].set_xlabel('$Chl-a$ in-situ $[\\text{mg}\\text{m}^{-3}]$')
    axs[0,0].set_ylabel('$Chl-a$ computed $[\\text{mg}\\text{m}^{-3}]$')
    axs[0,0].legend()
    axs[0,0].text(-0.1,1.05,'(a)',transform = axs[0,0].transAxes,fontsize='20')

    plot_one_line(axs[1,0],data['kd_442'],data['kd_VAE_output_442'],color='#CE2864',marker='x',label = 'SGVB')
    plot_one_line(axs[1,0],data['kd_442'],data['kd_log_output_442'],color='#1E88E5',marker='o',label = 'MAP AM param')
    #plot_one_line(axs[1,0],data['kd_442'],data['kd_NN_output_442'],color='green',marker='x',label = 'Neural Network')
    plot_one_line(axs[1,0],data['kd_442'],data['kd_logNN_output_442'],color='#674E03',marker='*',label = 'MAP SGVB param')
    data_1 = plot_one_line(axs[1,0],data['kd_442'],data['kd_un_output_442'],color='#004D40',marker='v',label = 'MAP Unperturbed param')
    axs[1,0].plot(np.linspace(np.min(data_1),np.max(data_1),20),np.linspace(np.min(data_1),np.max(data_1),20),alpha=0.3,lw=3,label='Perfect Linear Correlation',color ='gray',zorder=50)

    axs[1,0].set_xlabel('$kd_{442}$ in-situ $[m^{-1}$]')
    axs[1,0].set_ylabel('$kd_{442}$ computed $[m^{-1}$]')
    axs[1,0].legend()
    axs[1,0].text(-0.1,1.05,'(b)',transform = axs[1,0].transAxes,fontsize='20')
    
    plot_one_line(axs[0,1],data['bbp_442'],data['bbp_VAE_output_442'],color='#CE2864',marker='x',label = 'SGVB')
    plot_one_line(axs[0,1],data['bbp_442'],data['bbp_log_output_442'],color='#1E88E5',marker='o',label = 'MAP AM param')
    #plot_one_line(axs[0,1],data['bbp_442'],data['bbp_NN_output_442'],color='green',marker='x',label = 'Neural Network')
    plot_one_line(axs[0,1],data['bbp_442'],data['bbp_logNN_output_442'],color='#674E03',marker='*',label = 'MAP with SGVB param')
    data_1 = plot_one_line(axs[0,1],data['bbp_442'],data['bbp_un_output_442'],color='#004D40',marker='v',label = 'MAP Unperturbed param')
    axs[0,1].plot(np.linspace(np.min(data_1),np.max(data_1),20),np.linspace(np.min(data_1),np.max(data_1),20),alpha=0.3,lw=3,label='Perfect Linear Correlation',color ='gray',zorder=50)

    axs[0,1].set_xlabel('$b_{b,p,442}$ in-situ $[m^{-1}]$')
    axs[0,1].set_ylabel('$b_{b,p,442}$ computed $[m^{-1}]$')
    axs[0,1].legend()
    axs[0,1].text(-0.1,1.05,'(c)',transform = axs[0,1].transAxes,fontsize='20')

    plot_one_line(axs[1,1],data['RRS_510'],data['RRS_VAE_output_510'],color='#CE2864',marker='x',label = 'SGVB')
    plot_one_line(axs[1,1],data['RRS_510'],data['RRS_log_output_510'],color='#1E88E5',marker='o',label = 'MAP AM param')
    #plot_one_line(axs[1,1],data['RRS_510'],data['RRS_NN_output_510'],color='green',marker='x',label = 'Neural Network2')
    plot_one_line(axs[1,1],data['RRS_510'],data['RRS_logNN_output_510'],color='#674E03',marker='*',label = 'MAP SGVB param')
    data_1 = plot_one_line(axs[1,1],data['RRS_510'],data['RRS_un_output_510'],color='#004D40',marker='v',label = 'MAP Unperturbed param')
    axs[1,1].plot(np.linspace(np.min(data_1),np.max(data_1),20),np.linspace(np.min(data_1),np.max(data_1),20),alpha=0.3,lw=3,label='Perfect Linear Correlation',color ='gray',zorder=50)

    axs[1,1].set_xlabel('$R_{RS,442}$ in-situ $[sr^{-1}]$')
    axs[1,1].set_ylabel('$R_{RS,442}$ computed $[sr^{-1}]$')
    axs[1,1].legend()
    axs[1,1].text(-0.1,1.05,'(d)',transform = axs[1,1].transAxes,fontsize='20')

    plt.show()

def plot_constants_1(perturbation_path = MODEL_HOME + '/plot_data/perturbation_factors' ,vae_name = 'perturbation_factors_history_VAE.npy'):
    
    perturbation_factors_history_NN = torch.tensor(np.load(perturbation_path + '/' +  vae_name)).to(torch.float32)
    perturbation_factors_history_lognormal = torch.tensor(np.load(perturbation_path + '/perturbation_factors_history_lognormal.npy')).to(torch.float32)
    constant = read_constants(file1='./cte_lambda.csv',file2='./cst.csv')

    constants_history_NN = perturbation_factors_history_NN[:-1,5:]* np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])
    print('.....',len(constants_history_NN))
    constants_history_lognormal = perturbation_factors_history_lognormal[:,5:]* np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])

    



    names = ['$d_{\\text{CDOM}}$ [$\\text{m}^2(\\text{mgCDOM})^{-1}$]','$S_{\\text{CDOM}}$ [nm]','$q_1$','$q_2$',\
             '$\Theta^{\\text{min}}_{\\text{chla}}$ [$\\text{mgChla}\\text{(mgC)}^{-1}$]','$\Theta^{\\text{0}}_{\\text{chla}}$  [$\\text{mgChla}\\text{(mgC)}^{-1}$]',\
             '$\\beta$ [$\\text{mmol}\\text{m}^{-2}\\text{s}^{-1}$]','$\sigma$  [$\\text{mmol}\\text{m}^{-2}\\text{s}^{-1}$]','$b_{b,\\text{NAP}}$']
    print((constants_history_lognormal[0,0] - constants_history_lognormal[-1,0]).numpy(),(constants_history_NN[0,0] - constants_history_NN[-1,0]).numpy())
    percentages = np.array([np.min( [np.abs((constants_history_lognormal[0,i] - constants_history_lognormal[-1,i]).numpy()),\
                                     np.abs((constants_history_NN[0,i] - constants_history_NN[-1,i]).numpy())]   )/np.max( [np.abs((constants_history_lognormal[0,i] - constants_history_lognormal[-1,i]).numpy()),\
                                                                                                                            np.abs((constants_history_NN[0,i] - constants_history_NN[-1,i]).numpy())]   ) for i in range(9)])
    print([names[i] if percentages[i]>0.5 else '' for i in range(len(percentages))])
    print(percentages[percentages>0.5],'relative change between them.........................')


    for i in range(len(names)):
        print(names[i],'&','{:.5f}'.format(constants_history_NN[-1][i].numpy()),'&','{:.5f}'.format(constants_history_lognormal[-1][i].numpy()),'\\\\')

    labs = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']

    fig,axs = plt.subplots(3,3,tight_layout=True)


    for i,ax in enumerate(axs.flat):
        ax.plot(np.arange(len(constants_history_NN)),constants_history_NN[:,i],'-',color='gray',label='NN',lw=2)

        ax.set_ylabel(names[i])
        if i >=6:
            ax.set_xlabel('Number of Iterations')
        else:
            pass
        ax.plot(np.arange(len(constants_history_lognormal)),constants_history_lognormal[:,i],'-',color='#377eb8',label='BM',lw=2)
        ax.legend()
        ax.text(-0.1,1.1,labs[i],transform = ax.transAxes,fontsize='10')

    plt.show()

def plot_constants_2(perturbation_path = '/Users/carlos/Documents/OGS_one_d_model/plot_data/perturbation_factors',vae_name = 'perturbation_factors_history_VAE.npy'):
    
    perturbation_factors_history_NN = torch.tensor(np.load(perturbation_path + '/'+vae_name)).to(torch.float32)[:400]
    perturbation_factors_history_lognormal = torch.tensor(np.load(perturbation_path + '/perturbation_factors_history_lognormal.npy')).to(torch.float32)[:400]
    constant = read_constants(file1='./cte_lambda.csv',file2='./cst.csv')

    labs = ['(a)','(b)','(c)']

    def plot_track_absortion_ph(ax_,constant,past_perturbation_factors_NN, past_perturbation_factors_lognormal,len_past = 300,cmap = plt.cm.Blues):

        lambdas = np.array([412.5,442.5,490,510,555])
        storical_absortion_ph_NN =   (   past_perturbation_factors_NN[:,0].reshape((len_past,1))*constant['absortion_PH'].numpy() ).T
        storical_absortion_ph_lognormal =   (   past_perturbation_factors_lognormal[:,0].reshape((len_past,1))*constant['absortion_PH'].numpy() ).T
        #print('a')
        #print((storical_absortion_ph_lognormal[-1] - storical_absortion_ph_NN[-1]))

        percentages = np.abs((   np.abs(storical_absortion_ph_lognormal.T[0] - storical_absortion_ph_lognormal.T[-1])   -   \
                                 np.abs(storical_absortion_ph_NN.T[0] - storical_absortion_ph_NN.T[-1])   )/((storical_absortion_ph_lognormal.T[0] - storical_absortion_ph_lognormal.T[-1])))
        print(np.abs(storical_absortion_ph_lognormal.T[0] - storical_absortion_ph_lognormal.T[-1]),np.abs(storical_absortion_ph_NN.T[0] - storical_absortion_ph_NN.T[-1]),\
              np.abs(storical_absortion_ph_lognormal.T[0] - storical_absortion_ph_lognormal.T[-1])   -   np.abs(storical_absortion_ph_NN.T[0] - storical_absortion_ph_NN.T[-1]) )
        print(percentages[percentages<1],'relative change between them.........................')


        ax_.plot(lambdas,storical_absortion_ph_NN[:,-1],color = 'gray',label = 'NN')
        ax_.plot(lambdas,storical_absortion_ph_lognormal[:,-1],color = '#377eb8', label = 'BM')
        ax_.plot(lambdas,storical_absortion_ph_lognormal[:,0],'--',color = 'gray', label = 'Original values',alpha=0.5)
        ax_.set_xticks(lambdas,['412.5','442.5','490','510','555'])
        ax_.set_xlabel('Wavelenght [nm]',fontsize=10)
        ax_.set_ylabel('$a_{PH}$ $[\\text{m}^2\\text{(mgChl)}^{-1}]$',fontsize=10)
        ax_.tick_params(axis='y', labelsize=10)
        ax_.tick_params(axis='x', labelsize=10)
    


    def plot_track_scattering_ph(ax_,constant,past_perturbation_factors_NN, past_perturbation_factors_lognormal,len_past = 300, cmap = plt.cm.Blues):

        lambdas = np.array([412.5,442.5,490,510,555])
        #print(constant)
        storical_scattering_ph_NN =   (   past_perturbation_factors_NN[:,1].reshape((len_past,1))*constant['linear_regression_slope_s']*lambdas\
                                   +   past_perturbation_factors_NN[:,2].reshape((len_past,1))*constant['linear_regression_intercept_s'] ).T
        storical_scattering_ph_lognormal =   (   past_perturbation_factors_lognormal[:,1].reshape((len_past,1))*constant['linear_regression_slope_s']*lambdas\
                                         +        past_perturbation_factors_lognormal[:,2].reshape((len_past,1))*constant['linear_regression_intercept_s'] ).T

        ax_.plot(lambdas,storical_scattering_ph_NN[:,-1],color = 'gray',label='NN')
        ax_.plot(lambdas,storical_scattering_ph_lognormal[:,-1],color = '#377eb8', label = 'BM')
        ax_.plot(lambdas,storical_scattering_ph_lognormal[:,0],'--',color = 'gray', label = 'Original values',alpha=0.5)
        ax_.set_xticks(lambdas,['412.5','445.5','490','510','555'])
        ax_.set_xlabel('Wavelenght [nm]',fontsize=10)
        ax_.set_ylabel('$b_{PH}$ $[\\text{m}^2\\text{(mgChl)}^{-1})$',fontsize=10)
        ax_.tick_params(axis='y', labelsize=10)
        ax_.tick_params(axis='x', labelsize=10)
    
    def plot_track_backscattering_ph(ax_,constant,past_perturbation_factors_NN, past_perturbation_factors_lognormal,len_past = 300,cmap = plt.cm.Blues):

        lambdas = np.array([412.5,442.5,490,510,555])
        storical_scattering_ph_NN =   (   past_perturbation_factors_NN[:,3].reshape((len_past,1))*constant['linear_regression_slope_b']*lambdas\
                                   +   past_perturbation_factors_NN[:,4].reshape((len_past,1))*constant['linear_regression_intercept_b'] ).T

        storical_scattering_ph_lognormal =   (   past_perturbation_factors_lognormal[:,3].reshape((len_past,1))*constant['linear_regression_slope_b']*lambdas\
                                        +   past_perturbation_factors_lognormal[:,4].reshape((len_past,1))*constant['linear_regression_intercept_b'] ).T

        percentages = np.abs((   np.abs(storical_scattering_ph_lognormal.T[0] - storical_scattering_ph_lognormal.T[-1])   -   np.abs(storical_scattering_ph_NN.T[0] - storical_scattering_ph_NN.T[-1])   )/((storical_scattering_ph_lognormal.T[0] - storical_scattering_ph_lognormal.T[-1])))
        print( np.abs(storical_scattering_ph_lognormal.T[0] - storical_scattering_ph_lognormal.T[-1]),np.abs(storical_scattering_ph_NN.T[0] - storical_scattering_ph_NN.T[-1]))
        print(percentages[percentages<1],'relative change between them.........................')
        
        ax_.plot(lambdas,storical_scattering_ph_NN[:,-1],color = 'gray',label = 'NN')
        ax_.plot(lambdas,storical_scattering_ph_lognormal[:,-1],color = '#377eb8', label = 'BM')
        ax_.plot(lambdas,storical_scattering_ph_lognormal[:,0],'--',color = 'gray', label = 'Original values',alpha=0.5)
        
        ax_.set_xticks(lambdas,['412.5','445.5','490','510','555'])
        ax_.set_xlabel('Wavelenght [nm]',fontsize=10)
        ax_.set_ylabel('$b_{b,PH}$ $[\\text{m}^2\\text{(mgChl)}^{-1}]$',fontsize=10)
        ax_.tick_params(axis='y', labelsize=10)
        ax_.tick_params(axis='x', labelsize=10)

        
    fig, axs = plt.subplots(ncols = 3, nrows = 1,width_ratios = [1/3,1/3,1/3])
    plot_track_absortion_ph(axs[0],constant,perturbation_factors_history_NN,perturbation_factors_history_lognormal,400)
    plot_track_scattering_ph(axs[1],constant,perturbation_factors_history_NN,perturbation_factors_history_lognormal,400)
    plot_track_backscattering_ph(axs[2],constant,perturbation_factors_history_NN,perturbation_factors_history_lognormal,400)
    
    axs[0].text(-0.1,1.05,labs[0],transform = axs[0].transAxes,fontsize='20')
    axs[1].text(-0.1,1.05,labs[1],transform = axs[1].transAxes,fontsize='20')
    axs[2].text(-0.1,1.05,labs[2],transform = axs[2].transAxes,fontsize='20')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.show()

        

#comparison_alphas()

#plt.close()

test_indexes,train_indexes = customTensorData(data_path='npy_data',which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853).test_indexes,\
    customTensorData(data_path='npy_data',which='train',per_day = False,randomice=True,one_dimensional = True,seed = 1853).train_indexes


plot_scaterplot(test_indexes,vae_path = MODEL_HOME + '/VAE_model/results_VAE_VAEparam')
plot_constants_1(vae_name = 'perturbation_factors_history_CVAE_two.npy')
plot_constants_2(vae_name = 'perturbation_factors_history_CVAE_two.npy')
print(asdfasdf)

plot_kd(input_data_path = MODEL_HOME + '/results_bayes_lognormal_logparam',\
                figname = MODEL_HOME + '/plot_data/kd_lognormal_data.pdf',save=True,date_init = datetime(year=2012,month=1,day=1),\
                statistics=False, num_cols = 1,labels_names=['In situ data','Bayesian MAP output and Uncertainty'],ylim=[0,0.31],figsize=(17,17),\
        third_data_path = MODEL_HOME + '/results_VAE_VAEparam')

plot_bbp(input_data_path = MODEL_HOME + '/results_bayes_lognormal_logparam',\
                figname = MODEL_HOME + '/plot_data/bbp_lognormal_data.pdf',save=True,date_init = datetime(year=2012,month=1,day=1),\
                statistics=False, num_cols = 1,labels_names=['In situ data','Bayesian MAP output and Uncertainty'],ylim=[],figsize=(17,12),\
         third_data_path = MODEL_HOME + '/results_VAE_VAEparam')

plot_chla(input_data_path = MODEL_HOME + '/results_bayes_lognormal_logparam',\
                figname = MODEL_HOME + '/plot_data/chla_lognormal_data.pdf',save=True,date_init = datetime(year=2012,month=1,day=1),\
                statistics=False, num_cols = 1,labels_names=['In situ data','Bayesian MAP output and Uncertainty'],ylim=[],figsize=(17,12),\
          third_data_path = MODEL_HOME + '/results_VAE_VAEparam')


plot_chla(input_data_path = MODEL_HOME + '/results_bayes_lognormal_logparam',\
                figname = MODEL_HOME + '/plot_data/chla_lognormal_data.pdf',save=True,date_init = datetime(year=2012,month=1,day=1),\
                statistics=False, num_cols = 1,labels_names=['In situ data','Bayesian MAP output and Uncertainty'],ylim=[],figsize=(17,12),\
          third_data_path = MODEL_HOME + '/results_VAE_VAEparam')


#plot_with_u(input_data_path = '/Users/carlos/Documents/surface_data_analisis/LastVersion/results_bayes_lognormal_logparam',\
#                figname = '/Users/carlos/Documents/surface_data_analisis/LastVersion/plot_data/bayes_lognormal_data.pdf',save=True,date_init = datetime(year=2012,month=1,day=1),\
#                statistics=False, num_cols = 2,labels_names=['In situ data','Bayesian MAP output and Uncertainty'])


data = data_dataframe('/Users/carlos/Documents/surface_data_analisis/LastVersion/npy_data')

data['NAP'] = np.nan
data['CDOM'] = np.nan
second_run = read_second_run(MODEL_HOME + '/results_bayes_lognormal_logparam',include_uncertainty=True,abr='log_output')
data = second_run.merge(data,how='right',on='date')
second_run = read_second_run(MODEL_HOME + '/results_bayes_lognormal_VAEparam',include_uncertainty=True,abr='logNN_output')
data = second_run.merge(data,how='right',on='date')
second_run = read_second_run(MODEL_HOME + '/results_bayes_lognormal_unperturbed',include_uncertainty=True,abr='un_output')
data = second_run.merge(data,how='right',on='date')
second_run = read_second_run(MODEL_HOME + '/results_NN_NNparam',include_uncertainty=False,abr='NN_output')
data = second_run.merge(data,how='right',on='date')
second_run = read_second_run(MODEL_HOME + '/results_VAE_VAEparam',include_uncertainty=True,abr='VAE_output')
data = second_run.merge(data,how='right',on='date')
data.sort_values(by='date',inplace=True)
del second_run

#print(data['bbp_log_output_490'].iloc[-131],np.exp(data['chla_log_output'].iloc[-131]),np.exp(data['NAP_log_output'].iloc[-131]),np.exp(data['CDOM_log_output'].iloc[-131]))
#print(data['bbp_VAE_output_490'].iloc[-131],np.exp(data['chla_VAE_output'].iloc[-131]),np.exp(data['NAP_VAE_output'].iloc[-131]),np.exp(data['CDOM_VAE_output'].iloc[-131]))
#print(data['bbp_490'].iloc[-131])
#print(lkjsdf)
plt.close()

lambdas_names = ['412','442','490','510','555']
lambdas_values = ['412.5','442.5','490','510','555']

columns = []
names = []
labels = []
indexes = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']

labels_names=['In situ data','Bayesian MAP output and Uncertainty']
columns.append(('chla','chla_log_output','delta_chla_log_output','chla_VAE_output'))
names.append('$\\text{Chl-a } [\\text{mg}\\text{m}^{-3}]$')
labels.append((*labels_names,*labels_names,'Generative Neural Network Output'))


plot_parallel(data,columns,names,labels,statistics = False,histogram=False,date_init = datetime(year=2012,month=1,day=1),shadow_error = True,num_cols=1,\
              figname = 'holi',fontsize=30,colors = 1,save=False,figsize = (20,8),indexes = indexes,ylim=[],legend_fontsize='15',s=10)

def p_rmse(a,b,exp=False):
    data_a = data[a].iloc[test_indexes]
    data_b = data[b].iloc[test_indexes]
    if exp == True:
            return  '{:.6f}'.format(np.sqrt( np.nanmean(     ((np.exp(data_a) - np.exp(data_b))**2) /np.exp(data_a)**2       ) ))
    else:
        return  '{:.6f}'.format(np.sqrt( np.nanmean(((data_a - data_b)**2)/data_a**2) ))

def p_correlation(a,b,exp=False):

    data_a = data[a].iloc[test_indexes]
    data_b = data[b].iloc[test_indexes]
    data_statistics_1 = (data_a[~data_b.isnull()][~data_a[~data_b.isnull()].isnull()])
    data_statistics_2 = (data_b[~data_b.isnull()][~data_a[~data_b.isnull()].isnull()])
    
    if exp == True:
        data_statistics_1 = np.exp(data_statistics_1)
        data_statistics_2 = np.exp(data_statistics_2)
        
    return '{:.5f}'.format(stats.pearsonr(data_statistics_1,data_statistics_2).statistic)

for function_ in [p_rmse,p_correlation]:

    lambdas = ['412','442','490','510','555']
    lambdas_name = ['412.5','442.5','490','510','555']

    for i,lamb in enumerate(lambdas):
        a = 'RRS_' + lamb
        b = 'output_' + lamb
        print(  '$R_{RS,' + lambdas_name[i] + '}$ &',function_(a,'RRS_un_' + b), '&',  function_(a,'RRS_log_' + b),'&', function_(a,'RRS_logNN_' + b),'&', function_(a,'RRS_VAE_' + b),'\\\\' )

    for i,lamb in enumerate(lambdas):
        a = 'kd_' + lamb
        b = 'output_' + lamb
        print(  '$k_{d,' + lambdas_name[i] + '}$ &',function_(a,'kd_un_' + b), '&',  function_(a,'kd_log_' + b),'&', function_(a,'kd_logNN_' + b),'&', function_(a,'kd_VAE_' + b),'\\\\' )

    for i,lamb in enumerate(lambdas):
        if i in [0,3]:
            continue
        a = 'bbp_' + lamb
        b = 'output_' + lamb
        print(  '$b_{b,p,' + lambdas_name[i] + '}$ &',function_(a,'bbp_un_' + b), '&',  function_(a,'bbp_log_' + b),'&', function_(a,'bbp_logNN_' + b),'&', function_(a,'bbp_VAE_' + b),'\\\\' )


    print(  'chla &',function_('chla','chla_un_output',exp=True), '&',  function_('chla','chla_log_output',exp=True),'&', function_('chla','chla_logNN_output',exp=True),'&', function_('chla','chla_VAE_output',exp=True),'\\\\' )
    print('\n\n\n')


