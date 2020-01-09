# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:41:00 2019

@author: jiayi
"""

#set environment to use GPU
import os
#os.environ["THEANO_FLAGS"] = "device=gpu0"

from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
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

###############TEST################

#######################################Training################################
WINDOW = 10
df_train = pd.read_csv("WADI_normal_train.csv")
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






