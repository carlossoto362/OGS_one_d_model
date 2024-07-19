

import pandas as pd
import numpy as np
import matplotlib as plt
import scipy


def read_kd_data(kd_data_path='./kd_data',lambdas=[510.0,412.5,442.5,490.0,555.0]):
    """
    reeds the kd data. The files are supposed to be csv files with columns
    ,510.0,412.5,442.5,490.0,555.0,chla,CDOM,NAP,loss

    This function is supposed to be used with a DataFrame of data, a DataFrame of dates, and a Dataframe of constants, loaded when importing firstModel.
    Please create the global variables,
    
    >>>data = reed_data()
    >>>data = data[data['lambda']!=670]
    >>>dates = data['date'].drop_duplicates()
    
    data is a pandas DataFrame with the columns
    date,lambda,RRS,E_dir,E_dif,zenit,PAR.
    dates is a pandas series, with the dates from data on it. 
    
    reed_result reads from all the files on the path, so make sure that the path has no other file than the results. Each result file has the data of one date, and has
    to be stored in a file named %Y-%m-%d.csv. Is meant to be used after storing the data from the save_results function. 

    returns a pandas DataFrame with the columns
    
    """
    kd_datas = []
    for lamb in lambdas:
        kd_datas.append(pd.read_csv(kd_data_path + "/Kd.{:.2f}BOUSSOLEFit_AntoineMethod.txt".format(lamb),sep='\t'))
        kd_datas[-1]['lambda'] = str(lamb)
    kd_data = pd.concat(kd_datas)
    kd_data['date'] = [datetime(year=int(kd_data.iloc[k]['yyyy']),month=int(kd_data.iloc[k]['mm']),day=int(kd_data.iloc[k]['dd']),\
                                hour = int(kd_data.iloc[k]['HH']), minute = int(kd_data.iloc[k]['MM']),\
                                second = int(kd_data.iloc[k]['SS'])) for k in range(len(kd_data.iloc[:]))]
    #kd_data['date'] = datetime(year = kd_data['yyyy'],month=kd_data['mm'],day=kd_data['dd'],hour=kd_data['HH'],minute=kd_data['MM'],second=kd_data['SS'])
    return kd_data


