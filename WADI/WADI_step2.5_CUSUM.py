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

#for linux
#from tensorflow.keras import backend as K
#from tensorflow.keras.models import load_model
#for windows
from keras import backend as K
from keras.models import load_model

import matplotlib.pyplot as plt
import keras
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
from WADI_step1_Find_y import GT



#Make a new input including moving windows (a single array into multi window array)
def windowArray(inputX,WINDOW):
    inputX_win = []
    for i in range(len(inputX)-WINDOW+1):
        singleWin = inputX[i:i+WINDOW]
        #singleWin = singleWin.values
        inputX_win.append(singleWin)
    inputX_final = np.array(inputX_win)
    return inputX_final

def CUSUM(y_actual,y_predicted,sensor_head,bias):
    df_cusum_SH = pd.DataFrame()
    df_cusum_SL = pd.DataFrame()
    for col in sensor_head:
        print("CUSUM of:",col)
        SL = 0
        SH = 0
        list_SH = []
        list_SL = []
        for i in range(len(y_actual)):
    #        SH = np.max((0,(y_predicted[i]-y_actual[i]-0.05*1)))
    #        SL = np.min((0,(y_predicted[i]-y_actual[i]+0.05*1)))
            SH = np.max((0,SH+y_predicted[col].iloc[i]-y_actual[col].iloc[i]-bias))
            SL = np.min((0,SL+y_predicted[col].iloc[i]-y_actual[col].iloc[i]+bias))
            list_SH.append(SH)
            list_SL.append(SL)
        name_H = str(col)+"_H"
        name_L = str(col)+"_L"
        df_cusum_SH[name_H]=list_SH
        df_cusum_SL[name_L]=list_SL
    df_cusum = pd.concat([df_cusum_SH, df_cusum_SL], axis=1)
    return df_cusum

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
        plot(df_cusum_win[ele],Y)

def plot(t1,t2):

    x1 = np.arange(len(t1))
    x2 = np.arange(len(t2))

    plt.figure(1)

    plt.subplot(211)
    plt.plot(x1, t1)

    plt.subplot(212)
    plt.plot(x2, t2)

    plt.show()


def CalculateDiff(df_cusum_win,cum_sen_head,dic_TH,data_y):
    y_calculate = [0]*len(df_cusum_win)
    for ele in cum_sen_head:
        print(ele)
        name = ele.split("_")[-1]
        for i in range(len(df_cusum_win[ele])):
            if name == "H":
                if float(df_cusum_win[ele].iloc[i]) > dic_TH[ele]:
                    y_calculate[i] = 1
            elif name == "L":
                if float(df_cusum_win[ele].iloc[i]) <dic_TH[ele]:
                    y_calculate[i] = 1

        #calculate the difference
        print('checking error...')
        f1 = f1_score(data_y,y_calculate,average='binary')
        precision = precision_score(data_y,y_calculate,average='binary')
        recall = recall_score(data_y,y_calculate,average='binary')
        print('testing precision, recall, f1')
        print(precision, recall, f1)

        plot(y_calculate,data_y)

    return  y_calculate,precision, recall, f1

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

def ModifyRatio(df_adv,df_x):
    sen_a = np.absolute(np.matrix(df_adv[sensor_head]-df_x[sensor_head]))
    sen_b = np.absolute(np.matrix(df_adv[sensor_head]+df_x[sensor_head]))
    all_a = np.absolute(np.matrix(df_adv-df_x))
    all_b = np.absolute(np.matrix(df_adv+df_x))

    diff_sen = sen_a.sum()/sen_b.sum()
    print("seseor modified:",diff_sen)
    diff = all_a.sum()/all_b.sum()
    print("overall modified:",diff)
    act_change = np.count_nonzero(np.absolute(np.matrix(np.around(df_adv[actuator_head])-np.around(df_x[actuator_head]))))
    print("changed # of actuators:",act_change)
    print("total # of actuators:",len(df_adv)*len(actuator_head))
    print("changed percentage:",act_change/(len(df_adv)*len(actuator_head)))

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
STATUS = "ALL"
WINDOW = 10
#Y_att = "WADI_attack_Y.csv"
#perturbation = 0.1


if STATUS == "ALL":
    NORMALfile = "WADI_normal_train.csv"
    ATTACKfile = "WADI_attack.csv"
    MODEL = 'WADI.hdf5'
    ######
    X_att = "WADI_X_ADV_ALL10.csv"
    Y_name = "WADI_Y_attack_ALL10.csv"

#    PREDICTEDy = 'WADI_PREDICTION_all.csv'#for original predicted y
#    PREDICTEDy_csv = 'WADI_PREDICTION_adv_sen0.1.csv' #for noised predicted y
#    GRADIENT = "WADI_GRADIENT_10.csv"
#    NOISE_sen = "WADI_X_ADV_SEN1.csv"
#    NOISE_all = "WADI_X_ADV_ALL1.csv"

    header = pd.read_csv(ATTACKfile).columns

    actuator_head_MV = []
    actuator_head_P = []
    for i in header:
        if "_MV_" in i:
            actuator_head_MV.append(i)
        elif "_P_" in i:
            actuator_head_P.append(i)
        elif "_LS_" in i:
            actuator_head_P.append(i)

    actuator_head = actuator_head_MV+actuator_head_P
    sensor_head = list(header)
    for i in actuator_head:
        sensor_head.remove(i)

features = len(header)

#model = load_model('loss_na_p1.hdf5')
model = load_model(MODEL)
data_x_win, data_y_comp,scaler,min_max_scaler = PreProcess()
df_YY_actual = pd.DataFrame(data_y_comp,columns = header)

#df_Y = pd.read_csv(Y_name)#Y
#Y = df_Y#[WINDOW-1:]

#################Prediction################################################################
#NormalPredict(model,PREDICTEDy)



################difference######################################
df_adv = pd.read_csv(X_att)
data_x = np.reshape(np.array(data_x_win),(len(data_x_win)*WINDOW,len(header)))
df_x = pd.DataFrame(data_x,columns = df_adv.columns)

ModifyRatio(df_adv,df_x)


#df_a = pd.read_csv("WADI_X_ADV_SEN1.csv")
#df_b = pd.read_csv("WADI_X_ADV_ALL1.csv")
#
#ModifyRatio(df_a,df_b)

#############Prediciton adv################################################################
print('STEP4-start predicting...')
adv = np.expand_dims(df_adv,axis = 0)
array_adv = np.reshape(adv,(int(len(df_adv)/WINDOW),WINDOW,df_adv.shape[-1]))
#array_adv = np.array(data_x_win)
predict_test = model.predict(array_adv)
predict_adv = pd.DataFrame(predict_test,columns = header)


#################Get T(Va)#####################
df_input_x = predict_adv
GT_Y = GT(predict_adv,Y_name,WINDOW,features)
Y = GT_Y
data_y = Y
##write and read
#predict_adv.to_csv(PREDICTEDy_csv,index=False)


##################CUSUM###################################################################
df_YY_actual = df_YY_actual
df_YY_predict = predict_adv
#df_YY_predict = pd.read_csv(PREDICTEDy_csv) #for adv cusum
#df_YY_predict = pd.read_csv(PREDICTEDy) #for original cusum


#Calculate model cusum
print("calculating cusum....")
df_cusum_win= CUSUM(df_YY_actual,df_YY_predict,sensor_head,0.1)
cum_sen_head,cum_act_head = CUSUMhead(sensor_head,actuator_head)

plotData(df_cusum_win[cum_sen_head])



dic_TH={"1_AIT_001_PV_L":-10,"1_AIT_002_PV_L":-200,"1_LT_001_PV_H":50,"2_DPIT_001_PV_H":20,"2_FIC_101_CO_H":150,"2_FIC_101_CO_L":-100,"2_FIC_101_PV_H":0.25,"2_FIC_101_PV_L":-10,"2_FIC_101_SP_H":200,"2_FIC_201_CO_L":-50,"2_FIC_301_PV_L":-10,"2_FIC_301_CO_H":500,"2_FIC_301_SP_H":250,"2_FIC_401_PV_H":0.4,"2_FIC_401_PV_L":-12,"2_FIC_401_CO_H":200,"2_FIC_501_CO_H":500,"2_FIC_501_PV_H":20,"2_FIC_601_PV_L":-5,"2_FIC_601_SP_H":100,"2_FIT_001_PV_L":-4,"2_PIT_001_PV_H":200,"2_FIT_003_PV_L":-40,"2_FQ_301_PV_L":-10,"2_FQ_401_PV_H":0.2,"2_PIC_003_CO_H":250,"2_PIC_003_CO_L":-50,"2_PIC_003_PV_H":3,"2_PIC_003_PV_L":-100,"2_PIT_002_PV_H":50,"2_PIT_003_PV_H":5,"2_PIT_003_PV_L":-100,"2A_AIT_002_PV_L":-2.5,"2_PIT_001_PV_L":-20,"3_AIT_004_PV_L":-4,"3_AIT_005_PV_H":100,"3_FIT_001_PV_H":150,"3_FIT_001_PV_L":-150,"3_LT_001_PV_H":150,"LEAK_DIFF_PRESSURE_L":-50,"LEAK_DIFF_PRESSURE_H":0.3,"TOTAL_CONS_REQUIRED_FLOW_L":-200}


#Calculate the difference
sens = dic_TH.keys()
Y_calculate,precision, recall, f1 = CalculateDiff(df_cusum_win,sens,dic_TH,Y)
cm = confusion_matrix(Y, Y_calculate)
print("cm:",cm)
print("acc:", (cm[0][0]+cm[1][1])/cm.sum())
print("fp:", cm[0][1]/(cm[0][0]+cm[0][1]))
print("fn:", cm[1][0]/(cm[1][0]+cm[1][1]))

##############check attacks##################################################################
attack_y, attack_pre = checkAtt(np.array(Y_calculate), np.array(Y))
print('attack caught:')
attack_pre_set = list(set(attack_pre))
#print(attack_y)
print(len(attack_pre)-len(attack_y))
print(len(attack_pre_set))
print(attack_pre_set)
print('attack caught accuracy:')
print(len(attack_pre_set)/len(attack_y))
print(X_att)













