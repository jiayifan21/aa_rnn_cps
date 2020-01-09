# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:28:13 2019

@author: jiayi
"""
import os
os.environ["THEANO_FLAGS"] = "device=gpu0"

import tensorflow as tf
import pandas as pd
import numpy as np
#import time
import keras
from keras import backend as K
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from step2_rules import RuleCheck_stage1,RuleCheck_stage1_fix
from step2_rules import RuleCheck_all, RuleCheck_all_fix




def DataCom(df_x_scale,df_adv_att):
    df_800 = df_x_scale
    df_adv_800 = df_adv_att[::10]
    
    x_1600 = pd.concat([df_adv_800,df_800])
    y_true_1600 = [1]*len(df_adv_800)+[0]*len(df_800)
    
    return x_1600,y_true_1600

def AddNoise(data_x_win,YY_win,model,test):
    y_pred = model.output
    y_true = K.variable(np.array(df_YY_actual.iloc[0]))
    #ten_x_scale = K.variable(np.array(data_x_win[0]))
    loss = keras.losses.mean_squared_error(y_true, y_pred)
    grads = K.gradients(loss,model.input)[0]
    x_adv = K.sign(grads)
    sess =K.get_session() 
    init = tf.global_variables_initializer()
    FLAGS = []
    FLAG2S = []
    adv = []
    #len(df_x_scale)
    for i in range(len(data_x_win)):  
        print(i)
        sess.run(init)
        adv_i = sess.run(x_adv, feed_dict={model.input:data_x_win[i],y_true:np.array(df_YY_actual.iloc[i])})
        
        df_grd_i = pd.DataFrame(adv_i,columns = header)
        df_x_i = pd.DataFrame(data_x_win[i],columns = header)
    
        df_adv_sen = df_x_i[sensor_head] + df_grd_i[sensor_head]*0.01
        df_adv_act_mv = df_x_i[actuator_head_MV] + df_grd_i[actuator_head_MV]*0.5
        df_adv_act_p = df_x_i[actuator_head_P] + df_grd_i[actuator_head_P]*1
        
        df_adv = pd.concat([df_adv_sen,df_adv_act_mv,df_adv_act_p],axis=1)
        df_adv = df_adv[header]
        df_adv = np.clip(df_adv,0,1)
        
        #adv.append(np.array(df_adv))
        
        ###############################################with rule#####################################
        
        check_data_x = scaler.inverse_transform(min_max_scaler.inverse_transform(data_x_win[i]))
        check_data_adv = scaler.inverse_transform(min_max_scaler.inverse_transform(df_adv))
    
        #REMMEBER TO CHANGE to stage1
        if test == "all":
            #print("check original data..........")
            y_original = RuleCheck_all(check_data_x,header)
            #print("check adv data..........")
            y_check = RuleCheck_all(check_data_adv,header)
        else:
            #print("check original data..........")
            y_original = RuleCheck_stage1(check_data_x,header)
            #print("check adv data..........")
            y_check = RuleCheck_stage1(check_data_adv,header)
    
        flag = False
        for ii in range(len(y_original)):
            if y_original[ii]==0 and y_check[ii]!=0:
                flag = True
                
        if flag==False:
            adv.append(np.array(df_adv))
            FLAG2S.append(flag)
        else:
            print("flag = true")
            print("fix adv data..........")
            data_adv_fix = RuleCheck_all_fix(check_data_adv,header)
            df_adv_scale = scaler.transform(data_adv_fix)
            df_adv_fix = min_max_scaler.transform(df_adv_scale)
            print("checing 2nd time...")
            y_check2 = RuleCheck_all(data_adv_fix,header)
            flag2 = 0
            for ii in range(len(y_original)):
                if y_original[ii]==0 and y_check2[ii]!=0:
                    flag2 = 1
            if flag2==0:
                adv.append(np.array(df_adv_fix))
            else:
                print("thread will be found by rules")
                adv.append(np.array(data_x_win[i]))
            FLAG2S.append(flag2)
            
        FLAGS.append(flag)
    return FLAGS,adv

#####################data##################
#Read data
df_tr = pd.read_csv("normal_p1.csv")

#Data standardization
scaler = preprocessing.StandardScaler().fit(df_tr)
data_stand = scaler.transform(df_tr)
min_max_scaler = preprocessing.MinMaxScaler()
data_train_scale = min_max_scaler.fit_transform(data_stand)

######################################

##############Add noise####################
WINDOW = 10
test = "one"#"all"#
#sensor_head = pd.read_csv("attack_x_sensor.csv").columns
#actuator_head = pd.read_csv("attack_x_actuator.csv").columns
#actuator_head_MV = []
#actuator_head_P = []
#for i in actuator_head:
#    if "MV" in i:
#        actuator_head_MV.append(i)
#    else:
#        actuator_head_P.append(i)
#        
        
sensor_head = ["FIT101","LIT101"]
actuator_head_MV = ["MV101"]
actuator_head_P = ["P101","P102"]
actuator_head = ["MV101","P101","P102"]

header = list(df_tr.columns)

model_name = 'Defence.hdf5'
model = load_model(model_name)

        
df_data_x = pd.read_csv("new_p1_adv.csv")
data_x_win = np.reshape(np.array(df_data_x),(int(len(df_data_x)/WINDOW),WINDOW,df_data_x.shape[-1]))

df_YY_actual = pd.DataFrame(np.array([1]*len(df_data_x)))
YY_win = np.reshape(np.array(df_YY_actual),(int(len(df_data_x)/WINDOW),WINDOW,df_YY_actual.shape[-1]))


adv_all = []
#df_sen = df_x_scale[sensor_head]
#df_act = df_x_scale[actuator_head]
header = df_data_x.columns 


#df_adv_try = pd.read_csv("test_adv.csv")


df_TRUE,adv = AddNoise(data_x_win,YY_win,model,test)
reshape_adv = np.reshape(np.array(adv),(len(adv)*WINDOW,len(header)))
df_adv = pd.DataFrame(reshape_adv,columns = df_tr.columns)
df_adv.to_csv("defence_adv_p1.csv",index = False)
df_TRUE.to_csv("defen_TorF_p1.csv",index = False)

