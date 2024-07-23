#!/usr/bin/env python

import numpy as np
import torch

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
            """
            print(((backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                    Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[0]/chla[0]).clone().detach().numpy())
            print((perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant))[0].clone().detach().numpy())
            print(( (backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) * \
                     Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant))[0]/chla[0] + (perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant))[0]).clone().detach().numpy())

            def bbp_test(chla,lambda_):
                bbp_555 = 0.3*chla**(0.62)
                return (0.002 + 0.01*(0.5 - 0.25*np.log(chla)/np.log(10))*(550/lambda_)**(0.5*np.log(chla)/np.log(10)-0.3))*bbp_555
            print(bbp_test(0.00001,np.array([412,442,490,510,555])))
            print(asdfads)
            """
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
            #print(backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant)[-131,2],chla[-131] ,Carbon(chla[-131],PAR[-131,2],perturbation_factors,tensor=tensor,constant = constant),perturbation_factors,scattering_NAP(lambda_,tensor=tensor,constant = constant)[-131,2],NAP[-131],(backscattering_ph(lambda_,perturbation_factors,tensor=tensor,constant = constant) *Carbon(chla,PAR,perturbation_factors,tensor=tensor,constant = constant) + perturbation_factors[13]*0.005 * scattering_NAP(lambda_,tensor=tensor,constant = constant) * NAP)[-131])
        
            #print(asdfasdf)
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
