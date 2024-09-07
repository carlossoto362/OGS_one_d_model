

import Forward_module as fm
import read_data_module as rdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import scipy
from scipy import stats
from torch.utils.data import DataLoader,random_split
import time
import sys
import seaborn as sb
from Forward_module import evaluate_model_class,Forward_Model,RRS_loss,OBS_loss
from read_data_module import customTensorData ,read_constants
from torch.utils.data import DataLoader
from multiprocessing.pool import Pool
import matplotlib.colors as mcolors
from CVAE_model_part_two import NN_second_layer
import os
import sys

if 'OGS_ONE_D_HOME_PATH' in os.environ:
    MODEL_HOME = HOME_PATH =  os.environ["OGS_ONE_D_HOME_PATH"]
else:
    
    print("Missing local variable OGS_ONE_D_HOME_PATH. \nPlease add it with '$:export OGS_ONE_D_HOME_PATH=path/to/ogs/one/d/model'.")
    sys.exit()

def compute_jacobians(Forward_Model_, X, chla_hat_mean,perturbation_factors, constant = None):
    parameters_eval = perturbation_factors
    evaluate_model = evaluate_model_class(Forward_Model_,X,constant=constant,which_parameters='perturbations',chla=chla_hat_mean)
        
    jacobian_rrs = torch.autograd.functional.jacobian(evaluate_model.model_der,inputs=(parameters_eval))
    jacobian_kd = torch.autograd.functional.jacobian(evaluate_model.kd_der,inputs=(parameters_eval))
    jacobian_bbp = torch.autograd.functional.jacobian(evaluate_model.bbp_der,inputs=(parameters_eval))

    

    return jacobian_rrs,jacobian_kd,jacobian_bbp

def sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
                            title='Sensitivity of the parameters near the AM solution',lims=[-1e-1,13]):

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


        xticks = ['${a_{phy}}$','$\delta_{b_{phy,T}}$','${b_{phy,Int}}$','${b_{b,phy,T}}$','${b_{b,phy,Int}}$','${d_{\\text{CDOM}}}$','${S_{\\text{CDOM}}}$','${Q_a}$','${Q_b}$',\
                  '${\Theta^{\\text{min}}_{\\text{chla}}}$','${\Theta^{\\text{0}}_{\\text{chla}}}$',\
                  '${\\beta}$','${\sigma}$','${b_{b,\\text{NAP}}}$']
    
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
        sb.boxplot(data=jacobian_rrs_dataframe,ax=axs[0],whis=(10, 90),color='#377eb8',fliersize=1,widths=0.3)
        sb.boxplot(data=jacobian_kd_dataframe,ax=axs[1],whis=(10, 90),color='#377eb8',fliersize=1,widths=0.3)
        sb.boxplot(data=jacobian_bbp_dataframe,ax=axs[2],whis=(10, 90),color='#377eb8',fliersize=1,widths=0.3)
    
    

        axs[0].axes.xaxis.set_ticklabels([])
        axs[0].set_ylabel('$ \partial( \log{ R_{RS} }) \partial (\log{\delta_i})^{-1} $',fontsize=15)
        axs[0].tick_params(axis='y', labelsize=15)
        axs[0].text(1-0.04,0.9,'(a)',transform = axs[0].transAxes,fontsize=15)
        axs[0].set_ylim(*lims)
        axs[0].set_yscale('symlog')

        
        axs[1].axes.xaxis.set_ticklabels([])
        axs[1].set_ylabel('$ \partial (\log{ kd }) \partial (\log{\delta_i})^{-1} $',fontsize=15)
        axs[1].tick_params(axis='y', labelsize=15)
        axs[1].text(1-0.04,0.9,'(b)',transform = axs[1].transAxes,fontsize=15)
        axs[1].set_ylim(*lims)
        axs[1].set_yscale('symlog')
    
        axs[2].tick_params(axis='x', labelsize=15)
        axs[2].set_ylabel('$ \partial (\log{ b_{b,p} }) \partial (\log{\delta_i})^{-1} $',fontsize=15)
        axs[2].tick_params(axis='y', labelsize=15)
        axs[2].text(1-0.04,0.9,'(c)',transform = axs[2].transAxes,fontsize=15)
        axs[2].set_ylim(*lims)
        axs[2].set_yscale('symlog')
        axs[2].text(-0.05,-0.15,'$\delta_i,i= $',transform = axs[2].transAxes,fontsize=15)
        
        plt.show()


class evaluate_model():

    def __init__(self,data_path = MODEL_HOME + '/npy_data',iterations=10, my_device = 'cpu'):

        self.data = customTensorData(data_path=data_path,which='train',per_day = False,randomice=False,seed=1853)
        self.dates = self.data.dates
        self.my_device = 'cpu'
    
        self.constant = read_constants(file1=MODEL_HOME + '/cte_lambda.csv',file2=MODEL_HOME + '/cst.csv',my_device = my_device)
    
        self.x_a = torch.zeros(3)
        self.s_a = torch.eye(3)*100
        self.s_e = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2)#validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf

        self.lr = 0.029853826189179603
        self.batch_size = self.data.len_data
        self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False)
        self.model = Forward_Model(num_days=self.batch_size).to(my_device)

        self.loss = RRS_loss(self.x_a,self.s_a,self.s_e,num_days=self.batch_size,my_device = self.my_device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.iterations = iterations
        self.X,self.Y = next(iter(self.dataloader))

    def model_parameters_update(self,perturbation_factors):        
        self.model.perturbation_factors = perturbation_factors

    def model_chla_init(self,chla_hat):
        state_dict = self.model.state_dict()
        
        state_dict['chparam'] = chla_hat
        self.model.load_state_dict(state_dict)

    def step(self,iteration):
        RRS = self.model(self.X[:,:,1:],constant = self.constant)
        loss = self.loss(self.X[:,:,0],RRS,self.model.state_dict()['chparam'])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        #self.scheduler.step(loss)
        
    def predict(self):
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        list(map(self.step,range(self.iterations)))

        
    def return_chla(self):
        return self.model.state_dict()['chparam'].clone().detach()
    
class mcmc_class():
    def __init__(self,chla_hat,perturbation_factors = torch.ones(14),proposal_variance = 1,constant = None,num_iterations = 100):

        self.current_position = perturbation_factors + torch.randn(14)*0.01
        self.proposal_variance = proposal_variance
        self.model = evaluate_model()
        self.X, self.Y_nan  = next(iter(self.model.dataloader))
        
        self.Y = torch.masked_fill(self.Y_nan,torch.isnan(self.Y_nan),0)
        self.model.model_parameters_update(self.current_position)
        self.model.model_chla_init(chla_hat)
        self.model.iterations = 250
        self.model.predict()
        self.model.iterations = 5
        self.evaluate_model = evaluate_model_class(model = None,X = self.X[:,:,1:],constant = constant)
        self.loss = OBS_loss()

        self.history = torch.empty((num_iterations,14))

    def forward_sampling(self):
        return self.proposal_variance * torch.randn(14) + self.current_position

    def likelihood(self,position,chla):
        kd = self.evaluate_model.kd_der(parameters_eval = chla, perturbation_factors_ = self.current_position)
        bbp = self.evaluate_model.bbp_der(parameters_eval = chla, perturbation_factors_ = self.current_position)
            
        pred = torch.empty((chla.shape[0],9))
        pred[:,0] = chla[:,0,0]
        pred[:,1:6] = torch.masked_fill(kd,torch.isnan(self.Y_nan[:,1:6]),0)
        pred[:,6:] = torch.masked_fill(bbp,torch.isnan(self.Y_nan[:,6:]),0)

        loss = self.loss(self.Y,pred,self.Y_nan)
        return -(0.5)*loss*pred.shape[0]
                
    def step(self,iteration):
            
        chla_current = self.model.return_chla()
        log_likelihood_current = self.likelihood(self.current_position,chla_current)

        new_state = self.forward_sampling()
        self.model.model_parameters_update(new_state)
        self.model.predict()
        chla_new = self.model.return_chla()
        log_likelihood_new = self.likelihood(new_state,chla_new)

        rho = (log_likelihood_new-log_likelihood_current)
        rho = torch.exp(rho)
        if rho >= 1:
            self.current_position = new_state
        elif torch.rand(1) < rho:
            self.current_position = new_state
        else:
            self.model.model_parameters_update(self.current_position)
            self.model.model_chla_init(chla_current)
        self.history[iteration] = self.current_position

def saving_mcmc(input_):
    j = input_[0]
    perturbation_factors = input_[1]
    constant = input_[2]
    chla_hat = input_[3]
        
    init_time = time.time()
    num_iterations = 3000

    mcmc = mcmc_class(chla_hat,perturbation_factors = perturbation_factors,proposal_variance = 0.002,constant = constant,num_iterations = num_iterations)
            
    resulting_mcmc = list(map(mcmc.step,np.arange(num_iterations)))
        
        
    np.save(MODEL_HOME + '/mcmc/runs2/run_' + str(j)+'.npy',mcmc.history.numpy())
    print(MODEL_HOME + '/mcmc/runs2/run_' + str(j)+'.npy saved, time', (time.time() - init_time) )


def plot_constants_2(perturbation_path = MODEL_HOME + '/plot_data/perturbation_factors',vae_name = 'perturbation_factors_history_CVAE_chla_centered.npy'):
    
    perturbation_factors_history_NN = np.load(perturbation_path + '/'+vae_name)[-300:].mean(axis=1)
    perturbation_factors_history_lognormal =np.load(perturbation_path + '/perturbation_factors_history_lognormal.npy')[-1]
    perturbation_factors_mean = np.load(perturbation_path + '/perturbation_factors_mean_mcmc.npy')
    perturbation_factors_std = np.load(perturbation_path + '/perturbation_factors_std_mcmc.npy')
    constant = read_constants(file1='./cte_lambda.csv',file2='./cst.csv')

    labs = ['(a)','(b)','(c)']

    def plot_track_absortion_ph(ax_,constant,past_perturbation_factors_NN, past_perturbation_factors_lognormal,perturbation_factors_mean,perturbation_factors_std,cmap = plt.cm.Blues):

        lambdas = np.array([412.5,442.5,490,510,555])
        original_values = constant['absortion_PH'].numpy()
        storical_absortion_ph_NN =    past_perturbation_factors_NN[0]*(constant['absortion_PH'].numpy())
        storical_absortion_ph_lognormal =    past_perturbation_factors_lognormal[0]*(constant['absortion_PH'].numpy())
        storical_absortion_ph_mean =    perturbation_factors_mean[0]*(constant['absortion_PH'].numpy())
        storical_absortion_ph_std =    perturbation_factors_std[0]*(constant['absortion_PH'].numpy())

        ax_.plot(lambdas,storical_absortion_ph_NN,color = '#377eb8',label = 'SGVB')
        ax_.plot(lambdas,original_values,'--',color = 'black', label = 'Original values',alpha=0.5)
        ax_.set_xticks(lambdas,['412.5','442.5','490','510','555'])
        ax_.set_xlabel('Wavelenght [nm]',fontsize=15)
        ax_.set_ylabel('$a_{phy}$ $[\\text{m}^2\\text{(mgChl)}^{-1}]$',fontsize=15)
        ax_.tick_params(axis='y', labelsize=15)
        ax_.tick_params(axis='x', labelsize=15)

        ax_.fill_between(lambdas, storical_absortion_ph_mean - storical_absortion_ph_std, storical_absortion_ph_mean + storical_absortion_ph_std,color='gray',zorder = 0.1,alpha=0.8,label='63% confidence interval')
        ax_.plot(lambdas,storical_absortion_ph_mean,color = 'gray',label = 'mcmc mean value')
    


    def plot_track_scattering_ph(ax_,constant,past_perturbation_factors_NN, past_perturbation_factors_lognormal,perturbation_factors_mean,perturbation_factors_std, cmap = plt.cm.Blues):

        lambdas = np.array([412.5,442.5,490,510,555])
        #print(constant)
        original_values = constant['linear_regression_slope_s']*lambdas + (constant['linear_regression_intercept_s'])
        storical_scattering_ph_NN =   past_perturbation_factors_NN[1]*(constant['linear_regression_slope_s'])*lambdas\
                                   +   past_perturbation_factors_NN[2]*(constant['linear_regression_intercept_s'])
        storical_scattering_ph_lognormal =   past_perturbation_factors_lognormal[1]*(constant['linear_regression_slope_s'])*lambdas\
                                   +   past_perturbation_factors_lognormal[2]*(constant['linear_regression_intercept_s'])
        storical_scattering_ph_mean =   perturbation_factors_mean[1]*(constant['linear_regression_slope_s'])*lambdas\
                                   +   perturbation_factors_mean[2]*(constant['linear_regression_intercept_s'])
        storical_scattering_ph_std =   perturbation_factors_std[1]*(constant['linear_regression_slope_s'])*lambdas\
                                   +   perturbation_factors_std[2]*(constant['linear_regression_intercept_s'])

        ax_.plot(lambdas,storical_scattering_ph_NN,color = '#377eb8',label='SGVB')
        ax_.plot(lambdas,original_values,'--',color = 'black', label = 'Original values',alpha=0.5)
        ax_.set_xticks(lambdas,['412.5','445.5','490','510','555'])
        ax_.set_xlabel('Wavelenght [nm]',fontsize=15)
        ax_.set_ylabel('$b_{phy}$ $[\\text{m}^2\\text{(mgChl)}^{-1})$',fontsize=15)
        ax_.tick_params(axis='y', labelsize=15)
        ax_.tick_params(axis='x', labelsize=15)

        ax_.fill_between(lambdas, storical_scattering_ph_mean - storical_scattering_ph_std, storical_scattering_ph_mean + storical_scattering_ph_std,color='gray',zorder = 0.1,alpha=0.8,label='63% confidence interval')
        ax_.plot(lambdas,storical_scattering_ph_mean,color = 'gray',label = 'mcmc mean value')
    
    def plot_track_backscattering_ph(ax_,constant,past_perturbation_factors_NN, past_perturbation_factors_lognormal,perturbation_factors_mean,perturbation_factors_std,cmap = plt.cm.Blues):

        lambdas = np.array([412.5,442.5,490,510,555])
        #print(constant)
        original_values = constant['linear_regression_slope_b']*lambdas + (constant['linear_regression_intercept_b'])
        storical_backscattering_ph_NN =   past_perturbation_factors_NN[3]*(constant['linear_regression_slope_b'])*lambdas\
                                   +   past_perturbation_factors_NN[4]*(constant['linear_regression_intercept_b'])
        storical_backscattering_ph_lognormal =   past_perturbation_factors_lognormal[3]*(constant['linear_regression_slope_b'])*lambdas\
                                   +   past_perturbation_factors_lognormal[4]*(constant['linear_regression_intercept_b'])
        storical_backscattering_ph_mean =   perturbation_factors_mean[3]*(constant['linear_regression_slope_b'])*lambdas\
                                   +   perturbation_factors_mean[4]*(constant['linear_regression_intercept_b'])
        storical_backscattering_ph_std =   perturbation_factors_std[3]*(constant['linear_regression_slope_b'])*lambdas\
                                   +   perturbation_factors_std[4]*(constant['linear_regression_intercept_b'])

        ax_.plot(lambdas,storical_backscattering_ph_NN,color = '#377eb8',label='SGVB')
        ax_.plot(lambdas,original_values,'--',color = 'black', label = 'Original values',alpha=0.5)
        ax_.set_xticks(lambdas,['412.5','445.5','490','510','555'])
        ax_.set_xlabel('Wavelenght [nm]',fontsize=20)
        ax_.set_ylabel('$b_{b,phy}$ $[\\text{m}^2\\text{(mgChl)}^{-1})$',fontsize=20)
        ax_.tick_params(axis='y', labelsize=20)
        ax_.tick_params(axis='x', labelsize=20)

        ax_.fill_between(lambdas, storical_backscattering_ph_mean - storical_backscattering_ph_std, storical_backscattering_ph_mean + storical_backscattering_ph_std,color='gray',zorder = 0.1,alpha=0.8,label='63% confidence interval')
        ax_.plot(lambdas,storical_backscattering_ph_mean,color = 'gray',label = 'mcmc mean value')

        
    fig, axs = plt.subplots(ncols = 3, nrows = 1,width_ratios = [1/3,1/3,1/3],layout='constrained')
    plot_track_absortion_ph(axs[0],constant,perturbation_factors_history_NN,perturbation_factors_history_lognormal,perturbation_factors_mean,perturbation_factors_std)
    plot_track_scattering_ph(axs[1],constant,perturbation_factors_history_NN,perturbation_factors_history_lognormal,perturbation_factors_mean,perturbation_factors_std)
    plot_track_backscattering_ph(axs[2],constant,perturbation_factors_history_NN,perturbation_factors_history_lognormal,perturbation_factors_mean,perturbation_factors_std)
    
    axs[0].text(-0.1,1.05,labs[0],transform = axs[0].transAxes,fontsize='20')
    axs[1].text(-0.1,1.05,labs[1],transform = axs[1].transAxes,fontsize='20')
    axs[2].text(-0.1,1.05,labs[2],transform = axs[2].transAxes,fontsize='20')

    axs[0].legend(fontsize="15")
    axs[1].legend(fontsize="15")
    axs[2].legend(fontsize="15")

    plt.show()

def correlation_matrix(num_runs,mcmc_runs,plot=True,table=False):
    correlation_lenght_use = 0
    for which in range(14):

        correlation = np.empty((num_runs,500))
        for i in range(num_runs):
            correlation[i] = autocorr(mcmc_runs[i,:,which],range(500))
        correlation = np.mean(correlation,axis=0)

        for correlation_lenght in range(500):
            if correlation[correlation_lenght]<0.4:
                break
        if correlation_lenght > correlation_lenght_use:
            correlation_lenght_use = correlation_lenght
    
    data = mcmc_runs[:,::correlation_lenght,:]

    mcmc_runs_ = np.empty((280,14))
    
    for i in range(14):
        mcmc_runs_[:,i] = data[:,:,i].flatten()

    mcmc_runs_dataframe = pd.DataFrame()
    ticks = ['${a_{PH}}$','${b_{phy,T}}$','${b_{phy,\\text{Int}}}$','${b_{b,phy,T}}$','${b_{b,phy,\\text{Int}}}$','${d_{\\text{CDOM}}}$','${S_{\\text{CDOM}}}$','${Q_a}$','${Q_b}$',\
                  '${\Theta^{\\text{min}}_{\\text{chla}}}$','${\Theta^{\\text{0}}_{\\text{chla}}}$',\
                  '${\\beta}$','${\sigma}$','${b_{b,\\text{NAP}}}$']
    for i,tick in enumerate(ticks):
        mcmc_runs_dataframe[tick] = mcmc_runs_[:,i]
    
    correlation_matrix = mcmc_runs_dataframe.corr()
    print(correlation_matrix)

    if plot == True:
    
        fig,ax = plt.subplots(layout='constrained')
        colors1 = plt.cm.Greys_r(np.linspace(0.,1, 128))
        colors2 = plt.cm.Blues(np.linspace(0., 1, 128))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
        sb.heatmap(correlation_matrix, cmap=mymap, annot=True,vmin=-1,ax=ax)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        plt.show()
        plt.close()
    if table == True:

        print("$\\delta_i,i=$ & "+" & ".join(ticks) + "\\\\\\hline")
        for i in range(14):
            print(ticks[i] + " & " + " & ".join(["{:.2f}".format(correlation_matrix[ticks[i]][ticks[j]]) for j in range(i+1)]) + "".join(["&"]*(13-i)) +  "\\\\")
        
    

    
def autocorr(x,lags):
        '''numpy.corrcoef, partial'''

        corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
        corr = [1. if np.isnan(c) else c for c in corr ]
        return np.array(corr)

def parameters_statistics(num_runs,mcmc_runs,names,correlation_lenght=280,plot=True,table=False):

    if plot==True:
        fig,axs = plt.subplots(ncols=3,nrows=3,layout='constrained')
        for which,ax in enumerate(axs.flatten()):

            data = mcmc_runs[:,::correlation_lenght,which].flatten()
    
            (mu, sigma) = scipy.stats.norm.fit(data)
            n, bins, patches = ax.hist(data,bins=40,density=True,edgecolor='black',color='#377eb8',alpha=0.6)

            ks = scipy.stats.kstest((data - data.mean())/data.std(),scipy.stats.norm.cdf)


            
            ax.plot(bins, scipy.stats.norm.pdf(bins,loc=mu,scale=sigma), linewidth=2,color='black',label='mean value: {:.4f}\nstd value: {:.4f}\nKS statistic: {:.4f}\np-value: {:.4f}'.format(mu,sigma,ks.statistic,ks.pvalue))

        
            ax.axvline(x=constant_values[which],label= 'original values: {:.4f}'.format(constant_values[which]),color='gray',linestyle='--')
            ax.axvline( x = constant_values[which]*perturbation_factors[5+which],linestyle='dashdot',label='AM result: {:.4f}'.format(constant_values[which]*perturbation_factors[5+which]),color='blue')
            ax.axvline( x = constant_values[which]*perturbation_factors_NN[5+which],linestyle='dashdot',label='CVAE result: {:.4f}'.format(constant_values[which]*perturbation_factors_NN[5+which]),color='#f781bf')
            ax.set_xlabel(names[which],fontsize='15')
            ax.text(-0.1,1.05,fig_labels[which],transform = ax.transAxes,fontsize='20')

            y_lims = ax.get_ylim()
            x_lims = ax.get_xlim()
            ax.fill_between(np.linspace(mu-sigma,mu+sigma,20), [y_lims[0]]*20, [y_lims[1]]*20,color='gray',zorder = 0.1,alpha=0.8,label='63% confidence interval')
            ax.set_ylim(y_lims)
            ax.tick_params(axis='y', labelsize=15)
            ax.tick_params(axis='x', labelsize=15)
            ax.set_xlim((x_lims[0] - (x_lims[1] - x_lims[0] )*0.3,x_lims[1]))
            ax.legend(fontsize="10")
        plt.show()
    if table == True:
        mu = np.empty(9)
        sigma = np.empty(9)
        ks_value = np.empty(9)
        ks_pvalue = np.empty(9)

        for which in range(9):
            data = mcmc_runs[:,::correlation_lenght,which].flatten()
            (mu[which], sigma[which]) = scipy.stats.norm.fit(data)
            ks = scipy.stats.kstest((data - data.mean())/data.std(),scipy.stats.norm.cdf)
            ks_value[which] = ks.statistic
            ks_pvalue[which] = ks.pvalue
            
        statistics = ['Original value','CVAE result','MCMC result','KS test for normality','KS p-value for normality']
        print("&" + "&".join(statistics) + '\\\\\\hline')

        for which in range(len(names)):
            print( names[which] + "&" + "{:.4f}&{:.4f} &{:.4f} $\\pm$ {:.4f}&{:.4f}&{:.4f}\\\\".format(constant_values[which],\
                                                                                                                 constant_values[which]*perturbation_factors_NN[5+which],\
                                                                                                                 mu[which] ,\
                                                                                                                 sigma[which] ,\
                                                                                                                 ks_value[which] ,\
                                                                                                               ks_pvalue[which]))


def mcmc():
    my_device = 'cpu'
    precision = torch.float32
    perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_AM_test.npy')[-1]).to(torch.float32)
    constant = read_constants(file1=MODEL_HOME + '/cte_lambda.csv',file2=MODEL_HOME + '/cst.csv',my_device = my_device)

    best_result_config = torch.load(MODEL_HOME + '/VAE_model/model_second_part_final_config.pt')
    
    number_hiden_layers_mean = best_result_config['number_hiden_layers_mean']
    dim_hiden_layers_mean = best_result_config['dim_hiden_layers_mean']
    dim_last_hiden_layer_mean = best_result_config['dim_last_hiden_layer_mean']
    alpha_mean = best_result_config['alpha_mean']
    number_hiden_layers_cov = best_result_config['number_hiden_layers_cov']
    dim_hiden_layers_cov = best_result_config['dim_hiden_layers_cov']
    dim_last_hiden_layer_cov = best_result_config['dim_last_hiden_layer_cov']
    alpha_cov = best_result_config['alpha_cov']
    x_mul = torch.tensor(best_result_config['x_mul']).to(precision).to(my_device)
    x_add = torch.tensor(best_result_config['x_add']).to(precision).to(my_device)
    y_mul = torch.tensor(best_result_config['y_mul']).to(precision).to(my_device)
    y_add = torch.tensor(best_result_config['y_add']).to(precision).to(my_device)

    model_NN = NN_second_layer(output_layer_size_mean=3,number_hiden_layers_mean = number_hiden_layers_mean,\
                           dim_hiden_layers_mean = dim_hiden_layers_mean,alpha_mean=alpha_mean,dim_last_hiden_layer_mean = dim_last_hiden_layer_mean,\
                           number_hiden_layers_cov = number_hiden_layers_cov,\
                           dim_hiden_layers_cov = dim_hiden_layers_cov,alpha_cov=alpha_cov,dim_last_hiden_layer_cov = dim_last_hiden_layer_cov,x_mul=x_mul,x_add=x_add,\
                           y_mul=y_mul,y_add=y_add,constant = constant,model_dir = MODEL_HOME + '/VAE_model').to(my_device)

    
    model_NN.load_state_dict(torch.load(MODEL_HOME + '/VAE_model/model_second_part_chla_centered.pt'))
    model_NN.eval()

    data = customTensorData(data_path=MODEL_HOME + '/npy_data',which='train',per_day = False,randomice=False,one_dimensional = True,seed = 1853,device=my_device,normilized_NN='scaling')
    
    X,Y = next(iter(DataLoader(data, batch_size=data.len_data, shuffle=False)))

    z_hat,cov_z,mu_z,kd_hat,bbp_hat,rrs_hat = model_NN(X) #we are working with \lambda as imput, but the NN dosent use it. 
    mu_z = mu_z* model_NN.y_mul[0] + model_NN.y_add[0]
    list(map(saving_mcmc,[(j,perturbation_factors,constant,mu_z.unsqueeze(1)) for j in range(20)]))
                


if __name__ == '__main__':

    #mcmc()


    
    data_dir = MODEL_HOME + '/npy_data'

    my_device = 'cpu'
    
    constant = read_constants(file1=data_dir + '/../cte_lambda.csv',file2=data_dir+'/../cst.csv',my_device = my_device)
    data = customTensorData(data_path=data_dir,which='all',per_day = True,randomice=False,one_dimensional = False,seed = 1853,device=my_device)
    dataloader = DataLoader(data, batch_size=len(data.x_data), shuffle=False)

    chla_hat = np.load( MODEL_HOME + '/results_bayes_lognormal_logparam/X_hat.npy')
    kd_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_logparam/kd_hat.npy')[:,[0,2,4,6,8]]
    bbp_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_logparam/bbp_hat.npy')[:,[0,2,4]]
    rrs_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_logparam/RRS_hat.npy')
    
    mu_z = torch.tensor(chla_hat[:,[0,2,4]]).unsqueeze(1)

    X,Y = next(iter(dataloader))


    model = Forward_Model(num_days=1,learning_chla = False,learning_perturbation_factors = False).to(my_device)
    perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_lognormal.npy')[-1]).to(torch.float32)
    perturbation_factors = torch.ones(14)
    #jacobian_rrs,jacobian_kd,jacobian_bbp = compute_jacobians(model,X,mu_z,perturbation_factors,constant=constant) 
    #np.save(MODEL_HOME + '/Jacobians/jacobian_rrs_initialparam_lognormalresults.npy',jacobian_rrs.numpy())
    #np.save(MODEL_HOME + '/Jacobians/jacobian_kd_initialparam_lognormalresults.npy',jacobian_kd.numpy())
    #np.save(MODEL_HOME + '/Jacobians/jacobian_bbp_initialparam_lognormalresults.npy',jacobian_bbp.numpy())

    
    
    jacobian_rrs = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_rrs_lognormalparam_lognormalresults.npy')) #drrs/depsiloni
    jacobian_kd = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_kd_lognormalparam_lognormalresults.npy'))
    jacobian_bbp = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_bbp_lognormalparam_lognormalresults.npy'))


        
    #sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
    #                        title='Sensitivity of the parameters near the AM solution',lims=[-1e-1,8])

    jacobian_rrs = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_rrs_initialparam_lognormalresults.npy')) #drrs/depsiloni
    jacobian_kd = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_kd_initialparam_lognormalresults.npy'))
    jacobian_bbp = np.abs(np.load(MODEL_HOME + '/Jacobians/jacobian_bbp_initialparam_lognormalresults.npy'))

    chla_hat = np.load( MODEL_HOME + '/results_bayes_lognormal_unperturbed/X_hat.npy')
    kd_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_unperturbed/kd_hat.npy')[:,[0,2,4,6,8]]
    bbp_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_unperturbed/bbp_hat.npy')[:,[0,2,4]]
    rrs_hat = np.load(MODEL_HOME + '/results_bayes_lognormal_unperturbed/RRS_hat.npy')

    perturbation_factors = torch.ones(14)
    
    #sensitivity_boxplot(jacobian_rrs,jacobian_kd,jacobian_bbp,rrs_hat,kd_hat,bbp_hat,perturbation_factors,X,\
    #                        title='Sensitivity of the parameters with the literature values')


    ################MCMC#############
    indexes = customTensorData(data_path=data_dir,which='all',per_day = True,randomice=True,one_dimensional = False,seed = 1853,device=my_device).train_indexes
    perturbation_factors = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_AM_test.npy')[-1]).to(torch.float32)
    perturbation_factors_NN = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_CVAE_final.npy')[-1]).to(torch.float32)
    perturbation_factors_NN = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy')[-300:]).to(torch.float32).mean(axis=0)
    perturbation_factors_NN_std = torch.tensor(np.load(MODEL_HOME + '/plot_data/perturbation_factors/perturbation_factors_history_CVAE_chla_centered.npy')[-300:]).to(torch.float32).std(axis=0)
    chla_hat = torch.tensor(np.load( MODEL_HOME + '/results_bayes_lognormal_logparam/X_hat.npy')[:,[0,2,4]]).to(torch.float32).unsqueeze(1)[indexes]


    constant_values = np.array([constant['dCDOM'],constant['sCDOM'],5.33,0.45,constant['Theta_min'],constant['Theta_o'],constant['beta'],constant['sigma'],0.005])


    
    num_runs = 40
    mcmc_runs = np.empty((num_runs,3000,14))


    for i in range(num_runs):
        mcmc_runs[i] = np.load(MODEL_HOME + '/mcmc/run_' + str(i)+'.npy')

    #mcmc_runs = mcmc_runs[:,2000:,:]
    mcmc_runs_mean, mcmc_runs_std = np.empty((14)),np.empty((14))


    #correlation_matrix(num_runs,mcmc_runs,plot=False,table=False)

    mcmc_runs = mcmc_runs[:,:,5:] * constant_values

    mcmc_runs_mean = np.mean(mcmc_runs,axis=0)
    mcmc_percentile_2_5 = np.percentile(mcmc_runs,2.5,axis=0)
    
    mcmc_percentile_98_5 = np.percentile(mcmc_runs,98.5,axis=0)
    mcmc_percentile_16 = np.percentile(mcmc_runs,16,axis=0)
    mcmc_percentile_84 = np.percentile(mcmc_runs,84,axis=0)
    
    
    fig_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    names = ['$d_{\\text{CDOM}}$ [$\\text{m}^2(\\text{mgCDOM})^{-1}$]','$S_{\\text{CDOM}}$ [nm]','$Q_a$','$Q_b$',\
             '$\Theta^{\\text{min}}_{\\text{chla}}$ [$\\text{mgChla}\\text{(mgC)}^{-1}$]','$\Theta^{\\text{0}}_{\\text{chla}}$  [$\\text{mgChla}\\text{(mgC)}^{-1}$]',\
             '$\\beta$ [$\\text{mmol}\\text{m}^{-2}\\text{s}^{-1}$]','$\sigma$  [$\\text{mmol}\\text{m}^{-2}\\text{s}^{-1}$]','$b_{b,\\text{NAP}}$']
    which = 3
    fig,axs = plt.subplots(ncols=2,nrows=1,layout='constrained',width_ratios=[3/4,1/4])

    axs[0].axhline( y = constant_values[which],linestyle='--',label='original value',color='black')
    print(mcmc_runs_mean)

    axs[0].axhline( y = mcmc_runs_mean[500:,which].mean(),linestyle='dashdot',label='mcmc relaxation mean',color='blue')
    axs[0].axhline( y = constant_values[which]*perturbation_factors_NN[5+which],linestyle='dashdot',label='CVAE result',color='#f781bf')
    
    

    ynew = scipy.ndimage.uniform_filter1d(mcmc_runs_mean[:,which], size=100)
    axs[0].plot(ynew,label='mcmc')

    ynew1 = scipy.ndimage.uniform_filter1d(mcmc_percentile_2_5[:,which], size=100)
    ynew2 = scipy.ndimage.uniform_filter1d(mcmc_percentile_98_5[:,which], size=100)
    axs[0].fill_between(range(3000), ynew1, ynew2,color='gray',zorder = 0.1,alpha=0.6,label = '95% confidence interval of mcmc')
    
    ynew1 = scipy.ndimage.uniform_filter1d(mcmc_percentile_16[:,which], size=100)
    ynew2 = scipy.ndimage.uniform_filter1d(mcmc_percentile_84[:,which], size=100)
    axs[0].fill_between(range(3000), ynew1, ynew2,color='#377eb8',zorder = 0.1,alpha=0.8,label = '63% confidence interval of mcmc')
    


    axs[0].set_xlim(0,3000)
    axs[0].text(-0.1,1.05,'(a)',transform = axs[0].transAxes,fontsize=20)
    axs[0].set_ylabel(names[which])
    axs[0].set_xlabel('Iterations')

    axs[0].legend()

    data = mcmc_runs[:,2000:,which]
    data = data[:,::280].flatten()
    data = data
    
    
    n,bins,patches = axs[1].hist(data,orientation='horizontal',bins=40,edgecolor='black',color='#377eb8',alpha=0.6,density=True)
    (mu, sigma) = scipy.stats.norm.fit(data)
    axs[1].plot(scipy.stats.norm.pdf(bins,loc=mu,scale=sigma),bins , linewidth=2,color='black',label='normal distribution\nmean: {:.3f}\nstd: {:.3f}'.format(mu,sigma))
    axs[1].legend()
    axs[1].text(-0.1,1.05,'(b)',transform = axs[1].transAxes,fontsize=20)
    axs[1].set_xlabel('Probability density')

    axs[1].set_yticks([])
    axs[0].set_ylim(*axs[1].get_ylim())
    plt.show()
    plt.close()
    mcmc_runs = mcmc_runs[:,2000:,:]


    parameters_statistics(num_runs,mcmc_runs,names,correlation_lenght=280,plot=False,table=True)
    
    plot_constants_2(perturbation_path = MODEL_HOME + '/plot_data/perturbation_factors',vae_name = 'perturbation_factors_history_CVAE_chla_centered.npy')
            
