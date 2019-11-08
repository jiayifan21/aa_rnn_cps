# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:41:00 2019

@author: jiayi
"""

#set environment to use GPU
import os
os.environ["THEANO_FLAGS"] = "device=gpu1"

from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix


#Make a new input including moving windows (a single array into multi window array)
def windowArray(inputX,WINDOW):
    inputX_win = [] 
    for i in range(len(inputX)-WINDOW+1):
        singleWin = inputX[i:i+WINDOW]
        #singleWin = singleWin.values
        inputX_win.append(singleWin)
    inputX_final = np.array(inputX_win)
    return inputX_final

#creat model with moving window
def create_model_win(WINDOW,input_data):
    input_shape = (WINDOW,input_data.shape[2])
    print ('Creating model...')
#    input_cell_length = 51 #change to 26 if use sensor data only
#    timestamp = input_length
    model = Sequential()
    #model.add(Embedding(input_dim = 188, output_dim = 50, input_length = input_length))
    model.add(LSTM(activation='relu',return_sequences=True, units=100,input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(activation='relu',units=100))
    #model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(input_data.shape[2]))

    print ('Compiling...')
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy','mean_absolute_percentage_error'])#'rmsprop'
    
    return model

def CUSUM_bu(y_actual,y_predicted):
    SL = 0
    SH = 0
    list_col_H = []
    list_col_L = []
    for i in y_actual.columns:
        name_H = str(i)+"_H"
        name_L = str(i)+"_L"
        list_col_H.append(name_H)
        list_col_L.append(name_L)
    df_cusum_SH = pd.DataFrame(columns = list_col_H)
    df_cusum_SL = pd.DataFrame(columns = list_col_L)
    for col in y_actual.columns:
        for i in range(len(y_actual)):
    #        SH = np.max((0,(y_predicted[i]-y_actual[i]-0.05*1)))
    #        SL = np.min((0,(y_predicted[i]-y_actual[i]+0.05*1)))
            SH = np.max((0,SH+(y_predicted[col].iloc[i]-y_actual[col].iloc[i]-np.std(y_actual[col]))))
            SL = np.min((0,SL+(y_predicted[col].iloc[i]-y_actual[col].iloc[i]+np.std(y_actual[col]))))
            name_H = str(col)+"_H"
            name_L = str(col)+"_L"
            df_cusum_SH.set_value(i,name_H,SH)
            df_cusum_SL.set_value(i,name_L,SL)
    df_cusum = pd.concat([df_cusum_SH, df_cusum_SL], axis=1)
    return df_cusum

def CUSUM(y_actual,y_predicted):
    df_cusum_SH = pd.DataFrame()
    df_cusum_SL = pd.DataFrame()
    for col in y_actual.columns:
        print("CUSUM of:",col)
        SL = 0
        SH = 0
        list_SH = []
        list_SL = []
        for i in range(len(y_actual)):
    #        SH = np.max((0,(y_predicted[i]-y_actual[i]-0.05*1)))
    #        SL = np.min((0,(y_predicted[i]-y_actual[i]+0.05*1)))
            SH = np.max((0,SH+y_predicted[col].iloc[i]-y_actual[col].iloc[i]-0.1))
            SL = np.min((0,SL+y_predicted[col].iloc[i]-y_actual[col].iloc[i]+0.1))
            list_SH.append(SH)
            list_SL.append(SL)
        name_H = str(col)+"_H"
        name_L = str(col)+"_L"
        df_cusum_SH[name_H]=list_SH
        df_cusum_SL[name_L]=list_SL
    df_cusum = pd.concat([df_cusum_SH, df_cusum_SL], axis=1)
    return df_cusum




def CalculateDiff(df_cusum_win,cum_sen_head,dic_TH):
    y_calculate = [0]*len(df_cusum_win)
    for ele in cum_sen_head:
        print(ele)
        name = ele.split("_")[0]
        for i in range(len(df_cusum_win[ele])):
            if float(df_cusum_win[ele].iloc[i]) > dic_TH[name+"_H"] or float(df_cusum_win[ele].iloc[i]) < dic_TH[name+"_L"]:
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
    
###############TEST################

#######################################Training################################
WINDOW = 10
df_train = pd.read_csv("WADI_normal.csv")
df_train = df_train.fillna(0)
#df_train.drop('Row',axis=1, inplace=True)
#df_train.drop('Normal/Attack',axis=1, inplace=True)
#df_train.drop(' Timestamp',axis=1, inplace=True)
df_tr = df_train#[::10]

#Data standardization
scaler = preprocessing.StandardScaler().fit(df_tr)
data_stand = scaler.transform(df_tr)
df_stand_scale = pd.DataFrame(data_stand, columns = df_tr.columns)
#data_stand = preprocessing.scale(df_tr)

##Data scale to 0-1
min_max_scaler = preprocessing.MinMaxScaler()
data_train_scale = min_max_scaler.fit_transform(data_stand)
df_train_scale = pd.DataFrame(data_train_scale, columns = df_tr.columns)

#Add window
data_tr_win = windowArray(data_train_scale,WINDOW)
tr_x = data_tr_win[:-1]
tr_y = data_train_scale[WINDOW:]

#Create the LSTM model
model = create_model_win(WINDOW,tr_x)
#Load training data into the model
hist = model.fit(tr_x, tr_y, batch_size=200, epochs=10, validation_split = 0.1)
model.save('WADI.hdf5')

#model = load_model('final_p1.hdf5')

print("start predicting...")
#YY_predict = model.predict(tr_x)
#YY_predict = pd.DataFrame(YY_predict,columns = df_train.columns)
YY_actual = pd.DataFrame(tr_y,columns = df_train.columns)

#results = model.evaluate(tr_x, tr_y, batch_size=50)
#print("Loss and accuracy:",results)

#precision,recall = check(YY_predict, YY)
#mse = mean_squared_error(YY_actual,YY_predict)

#Save the model
#model.save('loss_na_all.hdf5')

#Calculate model cusum
#df_cusum_win_tr= CUSUM(YY_actual,YY_predict)
#df_cusum_tr= CUSUM_self(df_stand_scale,YY_predict)




###################################CUSUM#################################
#model = load_model('mse_0.002675.hdf5')

print("reading attack file...")
df_data_x = pd.read_csv("new_p1_adv.csv")
df_data_x_act = pd.read_csv("attack_p1_x.csv")
df_data_y = pd.read_csv("Y.csv")
#df_data_y1 = pd.read_csv("invariants check_Y.csv")
#df_data_y2 = pd.read_csv("attack_p1_y.csv")
#df_data_y = pd.read_csv("attack_y_stage1.csv")

data_x = df_data_x_act#[::10]
data_y = df_data_y[WINDOW:]

#sensor_head = pd.read_csv("attack_x_sensor.csv").columns
#actuator_head = pd.read_csv("attack_x_actuator.csv").columns

sensor_head = ["FIT101","LIT101"]
actuator_head = ["MV101","P101","P102"]

#Data standardization
data_x_stand = scaler.transform(data_x)
df_x_stand_scale = pd.DataFrame(data_x_stand, columns = df_tr.columns)

##Data scale to 0-1
data_x_scale = min_max_scaler.fit_transform(data_x_stand)
df_x_scale = pd.DataFrame(data_x_scale,columns = data_x.columns)

#Add window
df_x_win = np.reshape(np.array(data_x),(len(data_y),WINDOW,data_x.shape[-1]))#windowArray(data_x_scale,WINDOW)
data_x_win = df_x_win[:-1]
data_y_comp = data_x_scale[WINDOW:]

#Prediction
print('start predicting...')
YY_predict_test = model.predict(data_x_win)
df_YY_predict = pd.DataFrame(YY_predict_test,columns = df_data_x.columns) 
df_YY_actual = pd.DataFrame(data_y_comp,columns = df_data_x.columns)
results_attack = model.evaluate(data_x_win,df_YY_actual,batch_size=50)

#precision,recall = check(YY_predict, YY)
#mse = mean_squared_error(df_YY_actual,df_YY_predict)


#Calculate model cusum
print("calculating cusum....")
df_cusum_win= CUSUM(df_YY_actual,df_YY_predict)
#df_cusum= CUSUM_self(df_x_scale,df_YY_predict)

cum_sen_head,cum_act_head = CUSUMhead(sensor_head,actuator_head)

plotData(df_cusum_win)


dic_TH={"LIT101_L":-2,"LIT101_H":50,"FIT101_L":-1.5,"FIT101_H":2,"MV101_L":-1,"MV101_H":5,"P101_H":10,"P101_L":-1.5,"P102_H":1.5,"P102_L":-2}

#['FIT101_H', 'LIT101_H', 'MV101_H', 'P101_H', 'P102_H', 'FIT101_L','LIT101_L', 'MV101_L', 'P101_L', 'P102_L']


#Calculate the difference 
y_calculate,precision, recall, f1 = CalculateDiff(df_cusum_win,cum_sen_head,dic_TH)
cm = confusion_matrix(data_y, y_calculate)
print("cm:",cm)






