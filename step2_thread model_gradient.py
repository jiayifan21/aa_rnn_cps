# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:06:40 2019

@author: jiayi
"""

#set environment to use GPU
import os
#os.environ["THEANO_FLAGS"] = "device=gpu1"

import tensorflow as tf
import pandas as pd
import numpy as np
#import time
import copy
#from tensorflow.keras import backend as K
#from tensorflow.keras.models import load_model
from keras import backend as K
from keras.models import load_model

import matplotlib.pyplot as plt
import keras
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import preprocessing
from step2_rules import RuleCheck_stage1,RuleCheck_stage1_fix
from step2_rules import RuleCheck_all, RuleCheck_all_fix
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
import datetime



#Make a new input including moving windows (a single array into multi window array)
def windowArray(inputX,WINDOW):
    inputX_win = []
    for i in range(len(inputX)-WINDOW+1):
        singleWin = inputX[i:i+WINDOW]
        #singleWin = singleWin.values
        inputX_win.append(singleWin)
    inputX_final = np.array(inputX_win)
    return inputX_final

def CUSUMhead(sensor_head,actuator_head):
    cum_sen_head = []
    cum_act_head = []
    for ele in sensor_head:
        cum_sen_head.append(str(ele)+"_H")
        cum_sen_head.append(str(ele)+"_L")

    for ele in actuator_head:
        cum_act_head.append(str(ele)+"_H")
        cum_act_head.append(str(ele)+"_L")
    return cum_sen_head,cum_act_head

def plotData(df_cusum_win):
    for ele in df_cusum_win:
        print(ele)
        plot(df_cusum_win[ele],data_y)

def plot(t1,t2):

    x1 = np.arange(len(t1))
    x2 = np.arange(len(t2))

    plt.figure(1)

    plt.subplot(211)
    plt.plot(x1, t1)

    plt.subplot(212)
    plt.plot(x2, t2)

    plt.show()


#check attacks
#state is the predicted y
def checkAtt(state,df_train_y):
    attackList_y = []
    attackList_pre = []
    a = 0
    for i in range(1,len(state)):
        if df_train_y[i-1] == 0 and df_train_y[i] == 1:
            a+=1
            attackList_y.append(a)

        if state[i-1] == 0 and state[i] == 1:
            attackList_pre.append(a)

    return attackList_y,attackList_pre

def PreProcess():
    df_train = pd.read_csv(NORMALfile)
    df_tr = df_train[4000:]

    #Data standardization
    scaler = preprocessing.StandardScaler().fit(df_tr)
    data_stand = scaler.transform(df_tr)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_train_scale = min_max_scaler.fit_transform(data_stand)

    print("STEP1-reading attack file...")
    df_data_x = pd.read_csv(ATTACKfile)
    data_x = df_data_x

    #Data standardization
    data_x_stand = scaler.transform(data_x)
    #Data scale to 0-1
    data_x_scale = min_max_scaler.fit_transform(data_x_stand)
    df_x_scale = pd.DataFrame(data_x_scale,columns = data_x.columns)

    #Add window
    df_x_win = windowArray(data_x_scale,WINDOW)
    data_x_win = df_x_win[:-1]
    data_y_comp = df_x_scale[WINDOW:]

    return data_x_win, data_y_comp,scaler,min_max_scaler

def GetGradient(data_x_win,model,perturbation):
    y_pred = model.output
    y_true = K.variable(np.array(df_YY_actual.iloc[0]))
    #ten_x_scale = K.variable(np.array(data_x_win[0]))
    loss = keras.losses.mean_squared_error(y_true, y_pred)
    grads = K.gradients(loss,model.input)[0]
    x_adv = K.sign(grads)
    sess =K.get_session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    x_adv_0 = x_adv[0]

    adv = []
    if len(Y) != len(data_x_win):
        print("WARNING!!!! Unequal length of X and Y")
    #len(df_x_scale)
    for i in range(len(data_x_win)):
        adv_i = sess.run(x_adv_0, feed_dict={model.input:[data_x_win[i]],y_true:np.array(df_YY_actual.iloc[i])})
        if i%1000 == 0:
            print(i)
            print(datetime.datetime.now().isoformat())


        df_grd_i = pd.DataFrame(adv_i,columns = header)
        adv.append(np.array(df_grd_i))

    return adv


def AddNoises(data_x_win,Y,GRADIENT,perturbation):
    adv_sen = []
    adv_all = []
    Y = np.array(Y)
    read_grad = pd.read_csv(GRADIENT)
    data_grad = np.reshape(np.array(read_grad),(len(data_x_win),WINDOW,len(header)))

    for i in range(len(data_x_win)):
        df_grd_i = pd.DataFrame(data_grad[i],columns = header)
        df_x_i = pd.DataFrame(data_x_win[i],columns = header)


        if Y[i] == 0:
            df_adv_sen = df_x_i[sensor_head] + df_grd_i[sensor_head]*perturbation
            df_adv_act_mv = df_x_i[actuator_head_MV] + df_grd_i[actuator_head_MV]*0.5
            df_adv_act_p = df_x_i[actuator_head_P] + df_grd_i[actuator_head_P]*1
        elif Y[i] == 1:
            df_adv_sen = df_x_i[sensor_head] - df_grd_i[sensor_head]*perturbation
            df_adv_act_mv = df_x_i[actuator_head_MV] - df_grd_i[actuator_head_MV]*0.5
            df_adv_act_p = df_x_i[actuator_head_P] - df_grd_i[actuator_head_P]*1
        else:
            print("WARNING2!!! Y includes unexpected value")

        df_adv_s = pd.concat([df_adv_sen,df_x_i[actuator_head_MV],df_x_i[actuator_head_P]],axis=1)
        df_adv_s = df_adv_s[header]
        df_adv_s = np.clip(df_adv_s,0,1)

        df_adv = pd.concat([df_adv_sen,df_adv_act_mv,df_adv_act_p],axis=1)
        df_adv = df_adv[header]
        df_adv = np.clip(df_adv,0,1)

        adv_sen.append(np.array(df_adv_s))
        adv_all.append(np.array(df_adv))

    return adv_sen,adv_all


def NormalPredict(model,PREDICTEDy):
    #Prediction
    print('STEP2-start predicting...')
    YY_predict_test = model.predict(data_x_win)
    df_YY_predict = pd.DataFrame(YY_predict_test,columns = header)

    #Write and read
    df_YY_predict.to_csv(PREDICTEDy,index = False)


###########################################load model#################################################
###########################################load model#################################################
###########################################load model#################################################
#STATUS = "P1"

WINDOW = 12
#Y_all = "Y.csv"
Y_att = "Y_attack.csv"
perturbation = 0.01

NORMALfile = "normal_all.csv"
ATTACKfile = "attack_x.csv"
MODEL = 'SWaT.hdf5'
PREDICTEDy = 'PREDICTION_all.csv'#for original predicted y
PREDICTEDy_csv = 'PREDICTION_adv_sen0.1.csv' #for noised predicted y
GRADIENT = "GRADIENT_10.csv"
NOISE_sen = "X_ADV_SEN1.csv"
NOISE_all = "X_ADV_ALL1.csv"

sensor_head = pd.read_csv("attack_x_sensor.csv").columns
actuator_head = pd.read_csv("attack_x_actuator.csv").columns
header = pd.read_csv(ATTACKfile).columns

actuator_head_MV = []
actuator_head_P = []
for i in actuator_head:
    if "MV" in i:
        actuator_head_MV.append(i)
    else:
        actuator_head_P.append(i)

#model = load_model('loss_na_p1.hdf5')
model = load_model(MODEL)
data_x_win, data_y_comp,scaler,min_max_scaler = PreProcess()
df_YY_actual = pd.DataFrame(data_y_comp,columns = header)

df_Y = pd.read_csv(Y_att)#Y
Y = df_Y[WINDOW:]

#################Prediction################################################################
#NormalPredict(model,PREDICTEDy)

##############Add noise#####################################################################
##Get gradient
#start_time = datetime.datetime.now().isoformat()
#adv = GetGradient(data_x_win,model,perturbation)
#end_time = datetime.datetime.now().isoformat()
#reshape_adv = np.reshape(np.array(adv),(len(adv)*WINDOW,len(header)))
#df_adv = pd.DataFrame(reshape_adv,columns = header)
#
##Write and read
#df_adv.to_csv(GRADIENT,index = False)
##df_TRUE.to_csv(NOISE_rules,index = False)

#Add noises
adv_sen,adv_all = AddNoises(data_x_win,Y,GRADIENT,perturbation)

reshape_adv_sen = np.reshape(np.array(adv_sen),(len(adv_sen)*WINDOW,len(header)))
df_adv_sen = pd.DataFrame(reshape_adv_sen,columns = header)
df_adv_sen.to_csv(NOISE_sen,index = False)

reshape_adv_all = np.reshape(np.array(adv_all),(len(adv_all)*WINDOW,len(header)))
df_adv_all = pd.DataFrame(reshape_adv_all,columns = header)
df_adv_all.to_csv(NOISE_all,index = False)














