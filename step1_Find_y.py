# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:01:33 2020

@author: jiayi
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def PreProcess():
    NORMALfile = "normal_all.csv"
    df_train = pd.read_csv(NORMALfile)
    df_tr = df_train[4000:]

    #Data standardization
    scaler = preprocessing.StandardScaler().fit(df_tr)
    data_stand = scaler.transform(df_tr)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_train_scale = min_max_scaler.fit_transform(data_stand)

    return scaler,min_max_scaler

scaler,min_max_scaler = PreProcess()

def plot(t1,t2):

    x1 = np.arange(len(t1))
    x2 = np.arange(len(t2))

    plt.figure(1)

    plt.subplot(211)
    plt.plot(x1, t1)

    plt.subplot(212)
    plt.plot(x2, t2)

    plt.show()



def GT(df_input_x,Y_name,WINDOW,features):
    df_ref_file = "sensor_threshold.xlsx"
    df_ref = pd.read_excel(df_ref_file,index_col=0)
#    df_input_scaled = pd.read_csv(df_input_file)
    df_input_data = scaler.inverse_transform(min_max_scaler.inverse_transform(df_input_x))
    df_input = pd.DataFrame(df_input_data,columns = df_input_x.columns)
    #df_input = df_input[4000:]
    #df_input = df_input.reset_index(drop=True)
    
    
#    a = df_input.min()
    #b=pd.DataFrame(columns = df_ref.columns)
    
    Y = [0]*len(df_input)
    
    for i in range(len(df_input)):
        for item in df_ref.columns:
            if df_input.at[i,item]>df_ref.at["H",item] or df_input.at[i,item]<df_ref.at["L",item]:
                print(item)
    #            print(df_input.at[i,item])
    #            print(df_ref.at["H",item])
    #            print(df_ref.at["L",item])
                Y[i] = 1
    
    plot(Y,Y)
    Y = pd.DataFrame(Y)
    Y.to_csv(Y_name,index = False)
    return Y

