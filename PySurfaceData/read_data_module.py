#!/usr/bin/env python

import torch
import numpy as np
import scipy
import pandas as pd
from scipy import stats

"""
classes:
 customTensorData()

functions:
 read_constants()
 transform_to_data_dataframe()
 add_run_to_dataframe()
"""

class customTensorData():

    def __init__(self,data_path='./npy_data',transform=None,target_transform=None,train_percentage = 0.9,from_where = 'left',randomice=False,specific_columns=None,which='train',\
                 seed=None,per_day=True,precision=torch.float32,one_dimensional = False,normilized_NN='z-score',log_normal=True,device='cpu'):
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

        self.dates_all = x_data[:,-1]
        self.init_all = x_data[:,22:25]  ###########using lognormal distribution
        
        self.init_column_names = ['chla_init','NAP_init','CDOM_init']
        self.init_all[self.init_all==0] = 1
        self.init_all = np.log(self.init_all)


        self.x_data = np.delete(x_data,[22,23,24,25],axis=1)

        def zenith_fit(x,a,b,c,d):
            return a*np.cos(b*x + c) + d
        popt,pcov = scipy.optimize.curve_fit(zenith_fit,self.dates_all[self.x_data[:,-2]!=0],self.x_data[:,-2][self.x_data[:,-2]!=0],p0=[20,(2*np.pi)/360,0,40])
        self.x_data[:,-2] = zenith_fit(self.dates_all,*popt)

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
            self.y_data[:,-1][self.y_data[:,-1] == 0] = np.nan

            if log_normal == True:
                self.y_data[:,0] = np.log(self.y_data[:,0]) ############using log normal distribution
            
            self.x_column_names = ['RRS_412','RRS_442','RRS_490','RRS_510','RRS_555','Edif_412','Edif_442','Edif_490','Edif_510',\
                               'Edif_555','Edir_412','Edir_442','Edir_490','Edir_510','Edir_555','lambda_412','lambda_442',\
                               'lambda_490','lambda_510','lambda_555','zenith','PAR']


        
        self.date_info = '''date indicating the number of days since the first of january of 2000.'''

        if self.one_dimensional == True:
            self.x_data = np.delete(self.x_data,[-3,-4,-5,-6,-7],axis=1)
            self.x_column_names = np.delete(self.x_column_names,[-3,-4,-5,-6,-7])

        
        if specific_columns != None:
            self.x_data = self.x_data[specific_columns]
            self.x_column_names = self.x_column_names[specific_columns]

        self.len_data = len(self.dates_all)
        self.indexes = np.arange(self.len_data)
        if randomice == True:
            if type(seed) != type(None):
                np.random.seed(seed)
            np.random.shuffle(self.indexes)

        if from_where == 'right':
            self.indexes = np.flip(self.indexes)

        self.train_indexes = self.indexes[:int(self.len_data * train_percentage)]
        self.test_indexes = self.indexes[int(self.len_data * train_percentage):]


        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.precision=precision

        self.which = which
        self.my_indexes = self.indexes
        if self.which.lower().strip() == 'train' :
            self.my_indexes = self.train_indexes
        elif self.which.lower().strip() == 'test':
            self.my_indexes = self.test_indexes
        self.len_data = len(self.my_indexes)
        self.dates = self.dates_all[self.my_indexes]
        self.init = self.init_all[self.my_indexes]


        
        self.x_std = np.nanstd(self.x_data[self.train_indexes],axis=0)
        self.y_std = np.nanstd(self.y_data[self.train_indexes],axis=0)
        self.x_mean = np.nanmean(self.x_data[self.train_indexes],axis=0)
        self.y_mean = np.nanmean(self.y_data[self.train_indexes],axis=0)

        self.x_max = np.nanmax(self.x_data[self.train_indexes],axis=0)
        self.y_max = np.nanmax(self.y_data[self.train_indexes],axis=0)
        self.x_min = np.nanmin(self.x_data[self.train_indexes],axis=0)
        self.y_min = np.nanmin(self.y_data[self.train_indexes],axis=0)

        self.normilized_NN = normilized_NN
        if self.normilized_NN == 'z-score':
            self.x_normilized = (np.delete(self.x_data,[-3,-4,-5,-6,-7],axis=1) - np.delete(self.x_mean,[-3,-4,-5,-6,-7]))/np.delete(self.x_std,[-3,-4,-5,-6,-7])
            self.y_normilized = (self.y_data - self.y_mean)/self.y_std
            self.x_mul = torch.tensor(self.x_std).to(self.precision)
            self.y_mul = torch.tensor(self.y_std).to(self.precision)
            self.x_add = torch.tensor(self.x_mean).to(self.precision)
            self.y_add = torch.tensor(self.y_mean).to(self.precision)

        elif self.normilized_NN == 'scaling':
            
            self.x_normilized = (self.x_data - self.x_min)/(self.x_max - self.x_min)
            self.y_normilized = (self.y_data - self.y_min)/(self.y_max - self.y_min)
            self.x_mul = torch.tensor(self.x_max - self.x_min).to(self.precision)
            self.y_mul = torch.tensor(self.y_max - self.y_min).to(self.precision)
            self.x_add = torch.tensor(self.x_min).to(self.precision)
            self.y_add = torch.tensor(self.y_min).to(self.precision)
            
        self.device = device

    def change_which(self,which,indexes = None):
        
        self.which = which
        self.my_indexes = self.indexes
        if self.which.lower().strip() == 'train' :
            self.my_indexes = self.train_indexes
        elif self.which.lower().strip() == 'test':
            self.my_indexes = self.test_indexes
        elif self.which.lower().strip() == 'custom':
            self.my_indexes = indexes

        self.len_data = len(self.my_indexes)
        self.dates = self.dates_all[self.my_indexes]
        self.init = self.init_all[self.my_indexes]
        
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
            if self.normilized_NN != None:

                label = torch.tensor(self.y_normilized[self.my_indexes][idx]).unsqueeze(0)
                image = torch.tensor(self.x_normilized[self.my_indexes][idx]).unsqueeze(0)

            else:
                label = torch.tensor(self.y_data[self.my_indexes][idx]).unsqueeze(0)
                image = torch.tensor(self.x_data[self.my_indexes][idx]).unsqueeze(0)
          
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.to(self.precision).to(self.device), label.to(self.precision).to(self.device)




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


def transform_to_data_dataframe(data_path,which='all'):
    data = customTensorData(data_path=data_path,which=which,per_day = False,randomice=False)
    
    dataframe = pd.DataFrame(columns = data.x_column_names + data.y_column_names + data.init_column_names)
    dataframe[data.x_column_names] = data.x_data
    dataframe[data.y_column_names] = data.y_data
    dataframe[data.init_column_names] = data.init
    dataframe['date'] = [datetime(year=2000,month=1,day=1) + timedelta(days=date) for date in data.dates]
    dataframe.sort_values(by='date',inplace=True)
    dataframe['NAP'] = np.nan
    dataframe['CDOM'] = np.nan
    return dataframe
    
def add_run_to_dataframe(second_run_path,include_uncertainty=False,abr='output',name_index = None, ignore_name = None):
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

if __name__ == '__main__':
    data_path = '/Users/carlos/Documents/OGS_one_d_model/npy_data'
    data = customTensorData(data_path=data_path,which='all',per_day = False,randomice=True,seed=1853)



    
