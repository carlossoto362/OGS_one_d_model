

import Forward_module as fm
import read_data_module as rdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
import scipy
from scipy import stats
from torch.utils.data import DataLoader,random_split
import time
import sys
import seaborn as sb



MODEL_HOME = '/Users/carlos/Documents/OGS_one_d_model'


def compute_jacobians(Forward_Model, X, chla_hat_mean,perturbation_factors, constant = None):
    parameters_eval = perturbation_factors
    evaluate_model = fm.evaluate_model_class(Forward_Model,X,constant=constant,which_parameters='perturbations',chla=chla_hat_mean)
        
    jacobian_rrs = torch.autograd.functional.jacobian(evaluate_model.model_der,inputs=(parameters_eval))
    jacobian_kd = torch.autograd.functional.jacobian(evaluate_model.kd_der,inputs=(parameters_eval))
    jacobian_bbp = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))

    

    return jacobian_rrs,jacobian_kd,jacobian_bbp
    


if __name__ == '__main__':

    
    data_dir = MODEL_HOME + '/npy_data'

    my_device = 'cpu'
    
    constant = rdm.read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    data = rdm.customTensorData(data_path=data_dir,which='all',per_day = True,randomice=False,one_dimensional = False,seed = 1853,device=my_device)
    dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)

    chla_hat = np.load( MODEL_HOME + '/results_bayes_lognormal_logparam/X_hat.npy')
    kd_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_logparam/kd_hat.npy')[:,[0,2,4,6,8]]
    bbp_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_logparam/bbp_hat.npy')[:,[0,2,4]]
    rrs_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_logparam/RRS_hat.npy')
    
    mu_z = torch.tensor(chla_hat[:,[0,2,4]]).unsqueeze(1)

    X,Y = next(iter(dataloader))


    model = fm.Forward_Model(num_days=1,learning_chla = False,learning_perturbation_factors = False).to(my_device)
    perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_lognormal.npy')[-1]).to(torch.float32)
    perturbation_factors = torch.ones(14)
    #jacobian_rrs,jacobian_kd,jacobian_bbp = compute_jacobians(model,X,mu_z,perturbation_factors,constant=constant) 
    #np.save(MODEL_HOME + '/Jacobians/jacobian_rrs_initialparam_lognormalresults.npy',jacobian_rrs.numpy())
    #np.save(MODEL_HOME + '/Jacobians/jacobian_kd_initialparam_lognormalresults.npy',jacobian_kd.numpy())
    #np.save(MODEL_HOME + '/Jacobians/jacobian_bbp_initialparam_lognormalresults.npy',jacobian_bbp.numpy())

    
    
    jacobian_rrs = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_rrs_lognormalparam_lognormalresults.npy')) #drrs/depsiloni
    jacobian_kd = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_kd_lognormalparam_lognormalresults.npy'))
    jacobian_bbp = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_bbp_lognormalparam_lognormalresults.npy'))

    def sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
                            title='Sensitivity of the parameters near the AM solution'):

        rrs_normal = rrs_hat.reshape((rrs_hat.shape[0],5,1))
        epsilons = (np.tile(np.repeat(1/perturbation_factors,5).reshape(14,5).T,(X.shape[0],1)).reshape((X.shape[0],5,14)))
        normalization = rrs_normal * epsilons
        jacobian_rrs = np.mean(jacobian_rrs/normalization,axis=1)

        kd_normal = kd_hat.reshape((kd_hat.shape[0],5,1))
        normalization = kd_normal * epsilons
        jacobian_kd = np.mean(jacobian_kd/normalization,axis=1)


        bbp_normal = bbp_hat.reshape((bbp_hat.shape[0],3,1))
        epsilons = (np.tile(np.repeat(1/perturbation_factors,3).reshape(14,3).T,(X.shape[0],1)).reshape((X.shape[0],3,14)))
        normalization = bbp_normal * epsilons
        jacobian_bbp = np.mean(jacobian_bbp/normalization,axis=1)


        xticks = ['$\delta_{a_{PH}}$','$\delta_{b_{PH,T}}$','$\delta_{b_{PH,Int}}$','$\delta_{b_{b,PH,T}}$','$\delta_{b_{b,PH,Int}}$','$\delta_{d_{\\text{CDOM}}}$','$\delta_{S_{\\text{CDOM}}}$','$\delta_{q_1}$','$\delta_{q_2}$',\
                  '$\delta_{\Theta^{\\text{min}}_{\\text{chla}}}$','$\delta_{\Theta^{\\text{0}}_{\\text{chla}}}$',\
                  '$\delta_{\\beta}$','$\delta_{\sigma}$','$\delta_{b_{b,\\text{NAP}}}$']
    
        jacobian_rrs_dataframe = pd.DataFrame()
        for i in range(14):
            jacobian_rrs_dataframe[xticks[i]] = jacobian_rrs[:,i]

        jacobian_kd_dataframe = pd.DataFrame()
        for i in range(14):
            jacobian_kd_dataframe[xticks[i]] = jacobian_kd[:,i]

        jacobian_bbp_dataframe = pd.DataFrame()
        for i in range(14):
            jacobian_bbp_dataframe[xticks[i]] = jacobian_bbp[:,i]


        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(15*0.7, 9*0.65),
                                layout="constrained")
        sb.boxplot(data=jacobian_rrs_dataframe,ax=axs[0],whis=(0, 100),color=".6")
        sb.boxplot(data=jacobian_kd_dataframe,ax=axs[1],whis=(0, 100),color=".6")
        sb.boxplot(data=jacobian_bbp_dataframe,ax=axs[2],whis=(0, 100),color=".6")
    
    

        axs[0].axes.xaxis.set_ticklabels([])
        axs[0].set_ylabel('$ (\\nabla_\delta RR_S)(RR_S \delta^{-1})^{-1} $',fontsize=15)
        axs[0].tick_params(axis='y', labelsize=15)
        axs[0].text(1-0.04,0.9,'(a)',transform = axs[0].transAxes,fontsize=15)
        
        axs[1].axes.xaxis.set_ticklabels([])
        axs[1].set_ylabel('$ (\\nabla_\delta kd)(kd\delta^{-1})^{-1} $',fontsize=15)
        axs[1].tick_params(axis='y', labelsize=15)
        axs[1].text(1-0.04,0.9,'(b)',transform = axs[1].transAxes,fontsize=15)
    
        axs[2].tick_params(axis='x', labelsize=15)
        axs[2].set_ylabel('$ (\\nabla_\delta b_{b,p})(b_{b,p} \delta^{-1})^{-1} $',fontsize=15)
        axs[2].tick_params(axis='y', labelsize=15)
        axs[2].text(1-0.04,0.9,'(c)',transform = axs[2].transAxes,fontsize=15)

        axs[0].set_title(title,fontsize=20)
        
        plt.show()
        
    sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
                            title='Sensitivity of the parameters near the AM solution')

    jacobian_rrs = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_rrs_initialparam_lognormalresults.npy')) #drrs/depsiloni
    jacobian_kd = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_kd_initialparam_lognormalresults.npy'))
    jacobian_bbp = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_bbp_initialparam_lognormalresults.npy'))

    chla_hat = np.load( MODEL_HOME + '/results_bayes_lognormal_unperturbed/X_hat.npy')
    kd_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_unperturbed/kd_hat.npy')[:,[0,2,4,6,8]]
    bbp_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_unperturbed/bbp_hat.npy')[:,[0,2,4]]
    rrs_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_unperturbed/RRS_hat.npy')

    perturbation_factors = torch.ones(14)
    
    sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
                            title='Sensitivity of the parameters with the literature values')
