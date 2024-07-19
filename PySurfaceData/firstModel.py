#!/usr/bin/env python

"""
Functions for the inversion problem of a bio-optical model.

As part of the National Institute of Oceanography, and Applied Geophysics, I'm working on an invertion problem. A detailed description can be found at
https://github.com/carlossoto362/firstModelOGS.
this model contains the functions required to reed the satellite data and process it, in order to obtain the constituents: (chl-a, CDOM, NAP), 
using the first introduced model. 

When importing these functions, you are also importing numpy, torch, datetime, pandas and os. At the same time, you are reading the csv's with the constants.


"""

import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from datetime import datetime, timedelta
import pandas as pd
import os

#reading the constants from two csv files, one with the constants dependent on lambda, and the other with
#the ones that do not depend on it.

def reed_constants(file1='cte_lambda.csv',file2='cst.csv'):
    """
    function that reads the constants stored in file1 and file2. 
    file1 has the constants that are dependent on lambda, is a csv with the columns
    lambda, absortion_w, scattering_w, backscattering_w, absortion_PH, scattering_PH, backscattering_PH.
    file2 has the constants that are independent of lambda, is a csv with the columns
    name,values.

    reed_constants(file1,file2) returns a dictionary with all the constants. To access the absortion_w for examplea, write 
    constant = reed_constants(file1,file2)['absortion_w']['412.5'].
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

constant = reed_constants(file1='cte_lambda.csv',file2='cst.csv')

def reed_data():
    """
    function that reads the data stored in SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS/surface.yyy-mm-dd_12-00-00.txt
    reed_data() returns a pandas DataFrame with all the data available on SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS.
    Each file has to be a csv with the columns
    lambda,RRS,E_dir,E_dif,zenith,PAR.
    
    """
    data = os.listdir('SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS')
    names = ['lambda','RRS','E_dir','E_dif','zenith','PAR']
    all_data = pd.DataFrame(columns=['date','lambda','RRS','E_dir','E_dif','zenith','PAR'])
    for d in data:
        date = datetime.strptime(d,'surface.%Y-%m-%d_12-00-00.txt')
        one_data = pd.read_csv('SURFACE_DATA_ONLY_SAT_UPDATED_CARLOS/' + d ,sep=' ',names=names)
        one_data['date'] = [date]*6
        all_data = pd.concat([all_data,one_data],ignore_index=True)
        
    return all_data.sort_values(by='date')


################Functions for the absortion coefitient####################
def absortion_CDOM(lambda_):
    """
    Function that returns the mass-specific absorption coefficient of CDOM, function dependent of the wavelength lambda. 

    """
    return constant['dCDOM']*np.exp(-constant['sCDOM']*(lambda_ - 450.))

def absortion_NAP(lambda_):
    """
    Mass specific absorption coefficient of NAP.
    See Gallegos et al., 2011.
    """
    return constant['dNAP']*np.exp(-constant['sNAP']*(lambda_ - 440.))

def absortion(lambda_,chla,CDOM,NAP):
    """
    Total absortion coeffitient.
    aW,λ (values used from Pope and Fry, 1997), aP H,λ (values averaged and interpolated from
    Alvarez et al., 2022).
    """
    return constant['absortion_w'][str(lambda_)] + constant['absortion_PH'][str(lambda_)]*chla + \
        absortion_CDOM(lambda_)*CDOM + absortion_NAP(lambda_)*NAP


##############Functions for the scattering coefitient########################
def Carbon(chla,PAR):
    """
    defined from the carbon to Chl-a ratio. 
    theta_o, sigma, beta, and theta_min constants (equation and values computed from Cloern et al., 1995), and PAR
    the Photosynthetically available radiation, obtained from the OASIM model, see Lazzari et al., 2021.
    """
    nominator = chla
    exponent = -(PAR - constant['beta'])/constant['sigma']
    denominator = (constant['Theta_o'] * np.exp(exponent)/(np.exp(exponent) + 1)) +  constant['Theta_min']
    return nominator/denominator



def scattering_NAP(lambda_):
    """
    NAP mass-specific scattering coefficient.
    eNAP and fNAP constants (equation and values used from Gallegos et al., 2011)
    """
    return constant['eNAP']*(550./lambda_)**constant['fNAP']

def scattering(lambda_,PAR,chla,NAP):
    """
    Total scattering coefficient.
    bW,λ (values interpolated from Smith and Baker, 1981,), bP H,λ (values used
    from Dutkiewicz et al., 2015)
    """
    return constant['scattering_w'][str(lambda_)] + constant['scattering_PH'][str(lambda_)] * Carbon(chla,PAR) + \
        scattering_NAP(lambda_) * NAP

#################Functions for the backscattering coefitient#############

def backscattering(lambda_,PAR,chla,NAP):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    """
    return constant['backscattering_w'][str(lambda_)] + constant['backscattering_PH'][str(lambda_)] * \
        Carbon(chla,PAR) + 0.005 * scattering_NAP(lambda_) * NAP



###############Functions for the end solution of the equations###########
#The final result is written in terms of these functions, see ...

def c_d(lambda_,zenith,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return (absortion(lambda_,chla,CDOM,NAP) + scattering(lambda_,PAR,chla,NAP))/np.cos(zenith*np.pi/180)

def F_d(lambda_,zenith,PAR,chla,NAP):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return (scattering(lambda_,PAR,chla,NAP) - constant['rd'] * backscattering(lambda_,PAR,chla,NAP))/\
        np.cos(zenith*np.pi/180.)

def B_d(lambda_,zenith,PAR,chla,NAP):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return  constant['rd']*backscattering(lambda_,PAR,chla,NAP)/np.cos(zenith*np.pi/180) 

def C_s(lambda_,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return (absortion(lambda_,chla,CDOM,NAP) + constant['rs'] * backscattering(lambda_,PAR,chla,NAP) )/\
        constant['vs']

def B_u(lambda_,PAR,chla,NAP):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return (constant['ru'] * backscattering(lambda_,PAR,chla,NAP))/constant['vu']

def B_s(lambda_,PAR,chla,NAP):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return (constant['rs'] * backscattering(lambda_,PAR,chla,NAP))/constant['vs']

def C_u(lambda_,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return (absortion(lambda_,chla,CDOM,NAP) + constant['ru'] * backscattering(lambda_,PAR,chla,NAP))/\
        constant['vu']

def D(lambda_,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return (0.5) * (C_s(lambda_,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM) + \
                    ((C_s(lambda_,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM))**2 -\
                     4. * B_s(lambda_,PAR,chla,NAP) * B_u(lambda_,PAR,chla,NAP) )**(0.5))

def x(lambda_,zenith,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM) - C_s(lambda_,PAR,chla,NAP,CDOM)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM)) +\
        B_s(lambda_,PAR,chla,NAP) * B_u(lambda_,PAR,chla,NAP)
    nominator = -(C_u(lambda_,PAR,chla,NAP,CDOM) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM)) * F_d(lambda_,zenith,PAR,chla,NAP) -\
        B_u(lambda_,PAR,chla,NAP) * B_d(lambda_,zenith,PAR,chla,NAP)

    return nominator/denominator

def y(lambda_,zenith,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    denominator = (c_d(lambda_,zenith,PAR,chla,NAP,CDOM) - C_s(lambda_,PAR,chla,NAP,CDOM)) * \
        (c_d(lambda_,zenith,PAR,chla,NAP,CDOM) + C_u(lambda_,PAR,chla,NAP,CDOM)) +\
        B_s(lambda_,PAR,chla,NAP) * B_u(lambda_,PAR,chla,NAP)
    nominator = (-B_s(lambda_,PAR,chla,NAP) * F_d(lambda_,zenith,PAR,chla,NAP) ) +\
        (-C_s(lambda_,PAR,chla,NAP,CDOM) + c_d(lambda_,zenith,PAR,chla,NAP,CDOM)) * B_d(lambda_,zenith,PAR,chla,NAP)

    return nominator/denominator

def C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return E_dif_o - x(lambda_,zenith,PAR,chla,NAP,CDOM) * E_dir_o

def r_plus(lambda_,PAR,chla,NAP,CDOM):
    """
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return B_s(lambda_,PAR,chla,NAP)/D(lambda_,PAR,chla,NAP,CDOM)

def E_u_o(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM):
    """
    This is the analytical solution of the bio-chemical model. (work not published.)
    E_dif_o is the diffracted irradiance in the upper surface of the sea, 
    E_dir_o is the direct irradiance in the upper surface of the sea, 
    lambda is the wavelength, zenith is the zenith angle, PAR is the photosynthetic available radiation, chla the
    chloropyl-a, NAP the Non Algal Particles, and CDOM the Colored dissolved Organic Mater.
    """
    return C_plus(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM) * r_plus(lambda_,PAR,chla,NAP,CDOM) +\
        y(lambda_,zenith,PAR,chla,NAP,CDOM) * E_dir_o


##########################from the bio-optical model to RRS(Remote Sensing Reflectance)##############################
#defining Rrs
#Q=5.33*np.exp(-0.45*np.sin(np.pi/180.*(90.0-Zenith)))

def Q_rs(zenith):
    """
    Empirical result for the Radiance distribution function, 
    equation from Aas and Højerslev, 1999, 
    """
    return 5.33*np.exp(-0.45*np.sin((np.pi/180.0)*(90.0-zenith)))

def Rrs_minus(Rrs):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    """
    return Rrs/(constant['T']+constant['gammaQ']*Rrs)

def Rrs_plus(Rrs):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    """
    return Rrs*constant['T']/(1-constant['gammaQ']*Rrs)

def Rrs_MODEL(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM):
    """
    Remote Sensing Reflectance.
    Aas and Højerslev, 1999.
    """
    Rrs = E_u_o(E_dif_o,E_dir_o,lambda_,zenith,PAR,chla,NAP,CDOM)  /  (   Q_rs(zenith)*(E_dir_o + E_dif_o)   )
    return Rrs_plus( Rrs )




#################################Putting all together#################################################
class MODEL(nn.Module):
    """
    Bio-Optical model plus corrections, in order to have the Remote Sensing Reflectance, in terms of the invertion problem. 
    MODEL(x) returns a tensor, with each component being the Remote Sensing Reflectance for each given wavelength. 
    if the data has 5 rows, each with a different wavelength, RRS will return a vector with 5 components.  RRS has tree parameters, 
    self.chla is the chlorophil-a, self.NAP the Non Algal Particles, and self.CDOM the Colored Dissolved Organic Mather. 
    According to the invention problem, we want to estimate them by making these three parameters have two constraints,
    follow the equations of the bio-optical model, plus, making the RRS as close as possible to the value
    measured by the satellite.

    The parameters are initialized with random values between 0 and 1.
    
    """
    def __init__(self):
        super().__init__()
        self.chla = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.NAP = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)
        self.CDOM = nn.Parameter(torch.ones((1,1), dtype=torch.float32), requires_grad=True)

    def forward(self,x):
        """
        x: pandas dataframe with columns [E_dif,E_dir,lambda,zenith,PAR].
        """
        Rrs = torch.empty(len(x),)
        for i in range(len(x)):
            Rrs[i,] = Rrs_MODEL(x['E_dif'].iloc[i],x['E_dir'].iloc[i],x['lambda'].iloc[i],x['zenith'].iloc[i],\
                            x['PAR'].iloc[i],self.chla,self.NAP,self.CDOM)
        return Rrs


    
def train_loop(data_i,model,loss_fn,optimizer,N):
    """
    The train loop evaluates the Remote Sensing Reflectance RRS for each wavelength>>>pred=model(data_i), evaluates the loss function
    >>>loss=loss_fn(pred,y), force the value of the parameters (chla,NAP,CDOM) to be positive, evaluates the gradient of RRS with respect
    to the parameters, >>>loss.backward(), modifies the value of the parameters according to the optimizer criterium, >>>optimizer.step(),
    sets the gradient of RRS to cero, and prints the loss for a given number of iterations. This procedure is performed N times. 
    After N iterations, it returns two lists with the evolution of the loss function and the last evaluation of the model. 
    
    data_i has to be a pandas DataFrame with columns
    """
    
    size = len(data_i)
    data_i = data_i.loc[:,data_i.columns!='date'].astype(float)
    ls_val=[]
    ls_count=[]

    criterium = 1
    criterium_2 = 0
    i=0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #file_ = open('results_imprubment.txt','w')
    while (((criterium >N ) & (i<4000))):# or criterium_2 < 100):
        y = data_i['RRS'].to_numpy()
        y = torch.tensor(y).float()
        pred = model(data_i)
        loss = loss_fn(pred,y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for p in model.parameters():
            p.data.clamp_(0)
        
        if i % 1 == 0:
            ls_val.append(loss.item())
            ls_count.append(i)
            #print(i,list(model.parameters()))
            if i != 0:
                criterium = ls_val[-2] - ls_val[-1]
        if criterium <=0:
            criterium_2+=1
        i+=1
        
        scheduler.step(loss)
    #file_.close()
    return ls_val,ls_count,pred




#plt.plot(ls_count,ls_val)
#plt.show()


    
    
