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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score



def DataCom(df_x_scale,df_adv_att):
    df_800 = df_x_scale
    df_adv_800 = df_adv_att[::10]
    
    x_1600 = pd.concat([df_adv_800,df_800])
    y_true_1600 = [1]*len(df_adv_800)+[0]*len(df_800)
    
    return x_1600,y_true_1600

#creat model
def create_model(input_data):
    input_dim = input_data.shape[1]
    print ('Creating model...')
#    input_cell_length = 51 #change to 26 if use sensor data only
#    timestamp = input_length
    model = Sequential()
    #model.add(Embedding(input_dim = 188, output_dim = 50, input_length = input_length))
    model.add(Dense(activation='relu',units=100,kernel_initializer='random_normal',input_dim=input_dim))
    model.add(Dense(activation='relu',units=100,kernel_initializer='random_normal'))
    model.add(Dense(activation='relu',units=100,kernel_initializer='random_normal'))
    model.add(Dense(1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])#'rmsprop'
    
    return model

def ModelTrain(df_x_scale,df_adv_att,model_name):
    df_800 = df_x_scale
    df_adv_800 = df_adv_att[::10]
    
    x_1600 = pd.concat([df_adv_800,df_800])
    y_true_1600 = [1]*len(df_adv_800)+[0]*len(df_800)

    #create DNN model
    model = create_model(x_1600)
    #Load training data into the model
    hist = model.fit(np.array(x_1600), np.array(y_true_1600), batch_size=100, epochs=20, validation_split = 0.1)
    model.save(model_name)
    y_pred_1600 = model.predict_classes(x_1600)
    
    
    #Model evaluation 
    evaluation = model.evaluate(x_1600, y_true_1600)
    print("****************DNN train***************")
    print("loss and accuracy:",evaluation)

    Y_pred = y_pred_1600
    Y_origin = y_true_1600
    f1 = f1_score(Y_origin,Y_pred,average='binary')
    precision = precision_score(Y_origin,Y_pred,average='binary')
    recall = recall_score(Y_origin,Y_pred,average='binary')
    print('testing precision, recall, f1')
    print(precision, recall, f1)
    
    cm = confusion_matrix(Y_origin, Y_pred)
    print("cm:",cm)


def ModelTest(df_x_scale,df_adv_att,model_name):
    df_800 = df_x_scale
    df_adv_800 = df_adv_att[::10]
    
    x_1600 = pd.concat([df_adv_800,df_800])
    y_true_1600 = [1]*len(df_adv_800)+[0]*len(df_800)

    #create DNN model
    model = load_model(model_name)
    y_pred_1600 = model.predict_classes(x_1600)
    
    
    #Model evaluation 
    evaluation = model.evaluate(x_1600, y_true_1600)
    print("****************DNN test***************")
    print("loss and accuracy:",evaluation)

    Y_pred = y_pred_1600
    Y_origin = y_true_1600
    f1 = f1_score(Y_origin,Y_pred,average='binary')
    precision = precision_score(Y_origin,Y_pred,average='binary')
    recall = recall_score(Y_origin,Y_pred,average='binary')
    print('testing precision, recall, f1')
    print(precision, recall, f1)
    
    cm = confusion_matrix(Y_origin, Y_pred)
    print("cm:",cm)

def RF(df_x_scale,df_adv_att,test_x,test_adv):
    
    X,y = DataCom(df_x_scale,df_adv_att)
    test_x,test_y = DataCom(test_x,test_adv)
    
    clf = RandomForestClassifier(n_estimators=10,random_state=0)
    clf.fit(X, y)
    
    clf.score(test_x,test_y)
    print("****************Random Forest***************")
    print("score:",clf.score(test_x,test_y))
    
    result_y = clf.predict(test_x)
    
    Y_pred = result_y
    Y_origin = test_y
    f1 = f1_score(Y_origin,Y_pred,average='binary')
    precision = precision_score(Y_origin,Y_pred,average='binary')
    recall = recall_score(Y_origin,Y_pred,average='binary')
    print('testing precision, recall, f1')
    print(precision, recall, f1)

def DataProcess(df_tr,header,ratio,ATTACK,ATTACK_adv):
    #Data standardization
    scaler = preprocessing.StandardScaler().fit(df_tr)
    data_stand = scaler.transform(df_tr)
    min_max_scaler = preprocessing.MinMaxScaler()
    data_train_scale = min_max_scaler.fit_transform(data_stand)
    
    data_x = pd.read_csv(ATTACK)
    
    #Data standardization
    data_x_stand = scaler.transform(data_x)
    df_x_stand_scale = pd.DataFrame(data_x_stand, columns = header)
    
    #Data scale to 0-1
    data_x_scale = min_max_scaler.fit_transform(data_x_stand)
    
    train_x = pd.DataFrame(data_x_scale,columns = header)[:ratio]
    train_adv = pd.read_csv(ATTACK_adv)[:ratio]
    
    test_x = pd.DataFrame(data_x_scale,columns = header)[ratio:]
    test_adv = pd.read_csv(ATTACK_adv)[ratio:]
    
    return train_x,train_adv,test_x,test_adv

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

#####################data##################
NORMAL = "normal_all.csv"
ATTACK = "attack_x.csv"
ATTACK_adv = "defence_adv_all.csv"
#DEFENSE_adv = "defence_adv_all.csv"
#Read data
df_tr = pd.read_csv(NORMAL)
header = list(df_tr.columns)
ratio = int(len(df_tr)*0)

train_x,train_adv,test_x,test_adv = DataProcess(df_tr,header,ratio,ATTACK,ATTACK_adv)
model_name = 'Defence_all.hdf5'


######################################

#ModelTrain(train_x,train_adv,model_name)
ModelTest(test_x,test_adv,model_name)
#RF(train_x,train_adv,test_x,test_adv)



################difference######################################
#sensor_head = []
#actuator_head = []
#df_adv = pd.read_csv("defence_adv_p1.csv")
#data_x = pd.read_csv("new_p1_adv.csv")
#df_x = pd.DataFrame(data_x,columns = df_adv.columns)
#
#ModifyRatio(df_adv,df_x)



