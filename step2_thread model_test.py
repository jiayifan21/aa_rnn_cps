# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:06:40 2019

@author: jiayi
"""

#set environment to use GPU
import os
os.environ["THEANO_FLAGS"] = "device=gpu1"

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
#from LSTM_for_SWaT import predict, plot, check
#from keras.optimizers import Adam
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import preprocessing
from step2_rules import RuleCheck_stage1,RuleCheck_stage1_fix
from step2_rules import RuleCheck_all, RuleCheck_all_fix
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance


#Make a new input including moving windows (a single array into multi window array)
def windowArray(inputX,WINDOW):
    inputX_win = []
    for i in range(len(inputX)-WINDOW+1):
        singleWin = inputX[i:i+WINDOW]
        #singleWin = singleWin.values
        inputX_win.append(singleWin)
    inputX_final = np.array(inputX_win)
    return inputX_final

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

def PreProcess(STATUS):
    if STATUS == "ALL":
        NORMALfile = "normal_all.csv"
        ATTACKfile = "attack_x.csv"
        MODEL = 'final_all.hdf5'

    if STATUS == "P1":
        NORMALfile = "normal_p1.csv"
        ATTACKfile = "attack_p1_x.csv"
        MODEL = 'final_p1.hdf5'


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

def AddNoise(data_x_win,YY_win,model,STATUS,scaler,min_max_scaler,Y,perturbation = 0.1):
    y_pred = model.output
    Y = np.array(Y)
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
    if len(Y) != len(data_x_win):
        print("WARNING!!!! Unequal length of X and Y")
    #len(df_x_scale)
    for i in range(10):#range(len(data_x_win)):
        print(i)
        sess.run(init)
        adv_i = sess.run(x_adv[0], feed_dict={model.input:[data_x_win[i]],y_true:np.array(df_YY_actual.iloc[i])})

        df_grd_i = pd.DataFrame(adv_i,columns = header)
        df_x_i = pd.DataFrame(data_x_win[i],columns = header)

        if Y[i] == 0:
            df_adv_sen = df_x_i[sensor_head] + df_grd_i[sensor_head]*perturbation
            df_adv_act_mv = df_x_i[actuator_head_MV]# + df_grd_i[actuator_head_MV]*0.5
            df_adv_act_p = df_x_i[actuator_head_P]# + df_grd_i[actuator_head_P]*1
        elif Y[i] == 1:
            df_adv_sen = df_x_i[sensor_head] - df_grd_i[sensor_head]*perturbation
            df_adv_act_mv = df_x_i[actuator_head_MV]# - df_grd_i[actuator_head_MV]*0.5
            df_adv_act_p = df_x_i[actuator_head_P]# - df_grd_i[actuator_head_P]*1
        else:
            print("WARNING2!!! Y includes unexpected value")

        df_adv = pd.concat([df_adv_sen,df_adv_act_mv,df_adv_act_p],axis=1)
        df_adv = df_adv[header]
        df_adv = np.clip(df_adv,0,1)

        adv.append(np.array(df_adv))

        if i%5000 == 0:
            reshape_adv = np.reshape(np.array(adv),((len(adv)*WINDOW),len(header)))
            df_adv = pd.DataFrame(reshape_adv,columns = header)
            name = str(i)+"_noiseSensor10.csv"
            df_adv.to_csv(name,index=False)



        ###############################################with rule#####################################
#
#        check_data_x = scaler.inverse_transform(min_max_scaler.inverse_transform(data_x_win[i]))
#        check_data_adv = scaler.inverse_transform(min_max_scaler.inverse_transform(df_adv))
#
#        #REMMEBER TO CHANGE to stage1
#        if STATUS == "ALL":
#            #print("check original data..........")
#            y_original = RuleCheck_all(check_data_x,header)
#            #print("check adv data..........")
#            y_check = RuleCheck_all(check_data_adv,header)
#        else:
#            #print("check original data..........")
#            y_original = RuleCheck_stage1(check_data_x,header)
#            #print("check adv data..........")
#            y_check = RuleCheck_stage1(check_data_adv,header)
#
#        flag = False
#        for ii in range(len(y_original)):
#            if y_original[ii]==0 and y_check[ii]!=0:
#                flag = True
#
#        if flag==False:
#            adv.append(np.array(df_adv))
#            FLAG2S.append(flag)
#        else:
#            print("flag = true")
#            print("fix adv data..........")
#            if STATUS == "ALL":
#                data_adv_fix = RuleCheck_all_fix(check_data_adv,header)
#            else:
#                data_adv_fix = RuleCheck_stage1_fix(check_data_adv,header)
#            df_adv_scale = scaler.transform(data_adv_fix)
#            df_adv_fix = min_max_scaler.transform(df_adv_scale)
#            print("checing 2nd time...")
#            if STATUS == "ALL":
#                y_check2 = RuleCheck_all(data_adv_fix,header)
#            else:
#                y_check2 = RuleCheck_stage1(data_adv_fix,header)
#            flag2 = 0
#            for ii in range(len(y_original)):
#                if y_original[ii]==0 and y_check2[ii]!=0:
#                    flag2 = 1
#            if flag2==0:
#                adv.append(np.array(df_adv_fix))
#            else:
#                print("thread will be found by rules")
#                adv.append(np.array(data_x_win[i]))
#            FLAG2S.append(flag2)
#
#        FLAGS.append(flag)
    return FLAGS,adv

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
Y_all = "Y.csv"
Y_att = "Y_attack.csv"

if STATUS == "ALL":
    NORMALfile = "normal_all.csv"
    ATTACKfile = "attack_x.csv"
    MODEL = "final_all.hdf5"#'SWaT.hdf5'
    PREDICTEDy = 'PREDICTION_all.csv'#for original predicted y
    PREDICTEDy_csv = 'PREDICTION_adv_sen0.1.csv' #for noised predicted y
    NOISEx = "X_adv_sen0.1.csv"
    NOISE_rules = "TorF_all_noRule.csv"


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

if STATUS == "P1":
    NORMALfile = "normal_p1.csv"
    ATTACKfile = "attack_p1_x.csv"
    MODEL = 'final_p1.hdf5'
    PREDICTEDy = 'PREDICTION_p1.csv'#'PREDICTION_loss_na_all_win10.csv'#
    NOISEx = "defence_adv_p1.csv"
    NOISE_rules = "TorF_p1.csv"
    PREDICTEDy_csv = 'PREDICTION_p1_adv.csv'

    header = pd.read_csv(ATTACKfile).columns
    sensor_head = ["FIT101","LIT101"]
    actuator_head_MV = ["MV101"]
    actuator_head_P = ["P101","P102"]
    actuator_head = ["MV101","P101","P102"]

#model = load_model('loss_na_p1.hdf5')
model = load_model(MODEL)
data_x_win, data_y_comp,scaler,min_max_scaler = PreProcess(STATUS)
df_YY_actual = pd.DataFrame(data_y_comp,columns = header)

df_Y = pd.read_csv(Y_att)#Y
Y = df_Y[WINDOW:]

#################Prediction################################################################
#NormalPredict(model,PREDICTEDy)

##############Add noise#####################################################################

df_TRUE,adv = AddNoise(data_x_win,df_YY_actual,model,STATUS,scaler,min_max_scaler,Y)
reshape_adv = np.reshape(np.array(adv),(len(adv)*WINDOW,len(header)))
df_adv = pd.DataFrame(reshape_adv,columns = header)

#Write and read
df_adv.to_csv(NOISEx,index = False)
df_TRUE.to_csv(NOISE_rules,index = False)


################difference######################################
df_adv = pd.read_csv(NOISEx)
data_x = np.reshape(np.array(data_x_win),(len(data_x_win)*WINDOW,len(header)))
df_x = pd.DataFrame(data_x,columns = df_adv.columns)

ModifyRatio(df_adv,df_x)

#############Prediciton adv################################################################
df_adv = pd.read_csv(NOISEx)
print('STEP4-start predicting...')
adv = np.expand_dims(df_adv,axis = 0)
array_adv = np.reshape(adv,(int(len(df_adv)/WINDOW),WINDOW,df_adv.shape[-1]))
predict_test = model.predict(array_adv)
predict_adv = pd.DataFrame(predict_test,columns = header)

#write and read
predict_adv.to_csv(PREDICTEDy_csv,index=False)

##################CUSUM###################################################################
df_YY_actual = df_YY_actual
df_YY_predict = pd.read_csv(PREDICTEDy_csv) #for adv cusum
#df_YY_predict = pd.read_csv(PREDICTEDy) #for original cusum


#Calculate model cusum
print("calculating cusum....")
df_cusum_win= CUSUM(df_YY_actual,df_YY_predict,sensor_head,0.1)
cum_sen_head,cum_act_head = CUSUMhead(sensor_head,actuator_head)

plotData(df_cusum_win[cum_sen_head])


if STATUS == "ALL":
    a = 20000
    dic_TH={"FIT101_H":16,"FIT101_L":-25,"LIT101_H":400,"LIT101_L":-400,"AIT202_H":1,"AIT203_L":-400,"FIT201_L":-10,"DPIT301_L":-200,"FIT301_L":-3.7,"LIT301_L":-0.01,"AIT401_H":280,"LIT401_L":-70,"AIT501_L":-1,"AIT502_L":-2300,"AIT503_L":-150,"AIT504_L":-1,"FIT501_H":a,"FIT502_H":a,"FIT503_H":a,"FIT504_H":a,"PIT501_H":a,"PIT502_H":a,"PIT503_H":a,"FIT601_H":0.6,"FIT601_L":-20}



#Calculate the difference
sens = dic_TH.keys()
Y_calculate,precision, recall, f1 = CalculateDiff(df_cusum_win,sens,dic_TH,Y)
cm = confusion_matrix(Y, Y_calculate)
print("cm:",cm)


##############check attacks##################################################################
attack_y, attack_pre = checkAtt(np.array(Y_calculate), np.array(Y))
print('attack caught:')
attack_pre_set = list(set(attack_pre))
print(attack_y)
print(len(attack_pre)-len(attack_y))
print(len(attack_pre_set))
print(attack_pre_set)
print('attack caught accuracy:')
print(len(attack_pre_set)/len(attack_y))













