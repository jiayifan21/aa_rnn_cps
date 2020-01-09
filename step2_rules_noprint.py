# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:05:46 2019

@author: jiayi
"""
#set environment to use GPU
import os
#os.environ["THEANO_FLAGS"] = "device=gpu0"

from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn import preprocessing
import numpy as np


def plot(t1,t2):

    x1 = np.arange(len(t1))
    x2 = np.arange(len(t2))

    plt.figure(1)

    plt.subplot(211)
    plt.plot(x1, t1)

    plt.subplot(212)
    plt.plot(x2, t2)

    plt.show()




###############PARTS TO IMPORT DATA###############
#df_data = pd.read_csv("attack_x.csv")
##df_data.drop('Normal/Attack',axis=1, inplace=True)
##df_data.drop(' Timestamp',axis=1, inplace=True)
#df_y_true = pd.read_csv("attack_p1_y.csv")
#################RULES CHECKING###################33
def RuleCheck_stage1(data,header):
#    print("checking...")
    df_rules = pd.read_excel("./checking rules/New_Rules.xlsx")
    df_data = pd.DataFrame(data, columns = header)
    y = [0]*len(df_data)

    #rule 0
    RULE = 0
    V1 = 'LIT101'
    V1_judge = 500
    V2 = "MV101"
    V2_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########


    #rule 1
    RULE = 1
    V1 = 'LIT101'
    V1_judge = 800
    V2 = "MV101"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 2
    RULE = 2
    V1 = 'LIT101'
    V1_judge = 250
    V2 = "P101"
    V2_judge = 1
    V3 = "P102"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########


    return y

def RuleCheck_stage1_fix(data,header):
    print("checking...")
    df_rules = pd.read_excel("./checking rules/New_Rules.xlsx")
    df_data = pd.DataFrame(data, columns = header)
    y = [0]*len(df_data)

    #rule 0
    RULE = 0
    V1 = 'LIT101'
    V1_judge = 500
    V2 = "MV101"
    V2_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")
     ########


    #rule 1
    RULE = 1
    V1 = 'LIT101'
    V1_judge = 800
    V2 = "MV101"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")
     ########



    #rule 2
    RULE = 2
    V1 = 'LIT101'
    V1_judge = 250
    V2 = "P101"
    V2_judge = 1
    V3 = "P102"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")
     ########


    return df_data

###############################################################################33
def RuleCheck_all(data,header):
    df_rules = pd.read_excel("./checking rules/New_Rules.xlsx")
    df_data = pd.DataFrame(data, columns = header)
    y = [0]*len(df_data)

    #rule 0
    RULE = 0
    V1 = 'LIT101'
    V1_judge = 500
    V2 = "MV101"
    V2_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########


    #rule 1
    RULE = 1
    V1 = 'LIT101'
    V1_judge = 800
    V2 = "MV101"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 2
    RULE = 2
    V1 = 'LIT101'
    V1_judge = 250
    V2 = "P101"
    V2_judge = 1
    V3 = "P102"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 3
    RULE = 3
    V1 = 'LIT301'
    V1_judge = 800
    V2 = "P101"
    V2_judge = 2
    V3 = "P102"
    V3_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge and df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 4
    RULE = 4
    V1 = 'LIT301'
    V1_judge = 1000
    V2 = "P101"
    V2_judge = 1
    V3 = "P102"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 5
    RULE = 5
    V1 = 'LIT301'
    V1_judge = 800
    V2 = "MV201"
    V2_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########


    #rule 6
    RULE = 6
    V1 = 'LIT301'
    V1_judge = 1000
    V2 = "MV201"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 7
    RULE = 7
    V1 = 'FIT201'
    V1_judge = 0.5
    V2 = "P201"
    V2_judge = 1
    V3 = "P202"
    V3_judge = 1
    V4 = "P204"
    V4_judge = 1
    V5 = "P206"
    V5_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge or df_data[V4].iloc[i] != V4_judge or df_data[V5].iloc[i] != V5_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 8
    RULE = 8
    V1 = 'AIT201'
    V1_judge = 260
    V2 = "FIT201"
    V2_judge = 0.5
    V3 = "P201"
    V3_judge = 1
    V4 = "P202"
    V4_judge = 1


    flag = 0
    r1_idx = np.where(df_data[V1] > V1_judge)[0]
    r1_idx_and = np.where(df_data[V2].iloc[r1_idx] > V2_judge)[0]
    r1_idx_t = r1_idx_and + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V3].iloc[i] != V3_judge or df_data[V4].iloc[i] != V4_judge:
                y[i] = 1
                flag = 1

     ########




    #rule 9
    RULE = 9
    V1 = 'AIT503'
    V1_judge = 260
    V2 = "FIT201"
    V2_judge = 0.5
    V3 = "P201"
    V3_judge = 1
    V4 = "P202"
    V4_judge = 1


    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_and = np.where(df_data[V2].iloc[r1_idx] > V2_judge)[0]
    r1_idx_t = r1_idx_and + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V3].iloc[i] != V3_judge or df_data[V4].iloc[i] != V4_judge:
                y[i] = 1
                flag = 1

     ########




    #rule 10
    RULE = 10
    V1 = 'AIT202'
    V1_judge = 6.95
    V2 = "P203"
    V2_judge = 1
    V3 = "P204"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########




    #rule 11
    RULE = 11
    V1 = 'AIT203'
    V1_judge = 500
    V2 = "P205"
    V2_judge = 1
    V3 = "P206"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] > V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########




    #rule 12
    RULE = 12
    V1 = 'AIT203'
    V1_judge = 420
    V2 = "FIT201"
    V2_judge = 0.5
    V3 = "AIT402"
    V3_judge = 250
    V4 = "P205"
    V4_judge = 2
    V5 = "P206"
    V5_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_and = np.where(df_data[V2].iloc[r1_idx] > V2_judge)[0]
    r1_idx_and2 = np.where(df_data[V3].iloc[r1_idx] > V3_judge)[0]
    r1_idx_t = r1_idx_and2 + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V4].iloc[i] != V4_judge and df_data[V5].iloc[i] != V5_judge:
                y[i] = 1
                flag = 1

     ########





    #rule 13
    RULE = 13
    V1 = 'AIT402'
    V1_judge = 250
    V2 = "P205"
    V2_judge = 1
    V3 = "P206"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########






    #rule 14
    RULE = 14
    V1 = 'LIT301'
    V1_judge = 250
    V2 = "P301"
    V2_judge = 1
    V3 = "P302"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########





    #rule 15
    RULE = 15
    V1 = 'LIT401'
    V1_judge = 1000
    V2 = "P301"
    V2_judge = 1
    V3 = "P302"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########




    #rule 16
    RULE = 16
    V1 = 'LIT401'
    V1_judge = 250
    V2 = "P401"
    V2_judge = 1
    V3 = "P402"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 17
    RULE = 17
    V1 = 'LIT401'
    V1_judge = 250
    V2 = "UV401"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 18
    RULE = 18
    V1 = 'FIT401'
    V1_judge = 0.5
    V2 = "UV401"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 19
    RULE = 19
    V1 = 'AIT402'
    V1_judge = 240
    V2 = "P403"
    V2_judge = 1
    V3 = "P404"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1

     ########



    #rule 20
    RULE = 20
    V1 = 'UV401'
    V1_judge = 1
    V2 = "P501"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] == V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########




    #rule 21
    RULE = 21
    V1 = 'FIT401'
    V1_judge = 0.5
    V2 = "P501"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########




    #rule 22
    RULE = 22
    V1 = 'LIT101'
    V1_judge = 1100
    V2 = "P601"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] > V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1

     ########


    return y


######################################################################################3
def RuleCheck_all_fix(data,header):
    df_rules = pd.read_excel("./checking rules/New_Rules.xlsx")
    df_data = pd.DataFrame(data, columns = header)
    y = [0]*len(df_data)

    #rule 0
    RULE = 0
    V1 = 'LIT101'
    V1_judge = 500
    V2 = "MV101"
    V2_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")

     ########


    #rule 1
    RULE = 1
    V1 = 'LIT101'
    V1_judge = 800
    V2 = "MV101"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")

     ########



    #rule 2
    RULE = 2
    V1 = 'LIT101'
    V1_judge = 250
    V2 = "P101"
    V2_judge = 1
    V3 = "P102"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")

     ########



    #rule 3
    RULE = 3
    V1 = 'LIT301'
    V1_judge = 800
    V2 = "P101"
    V2_judge = 2
    V3 = "P102"
    V3_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge and df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
#                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########



    #rule 4
    RULE = 4
    V1 = 'LIT301'
    V1_judge = 1000
    V2 = "P101"
    V2_judge = 1
    V3 = "P102"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########



    #rule 5
    RULE = 5
    V1 = 'LIT301'
    V1_judge = 800
    V2 = "MV201"
    V2_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")
     ########


    #rule 6
    RULE = 6
    V1 = 'LIT301'
    V1_judge = 1000
    V2 = "MV201"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")
     ########



    #rule 7
    RULE = 7
    V1 = 'FIT201'
    V1_judge = 0.5
    V2 = "P201"
    V2_judge = 1
    V3 = "P202"
    V3_judge = 1
    V4 = "P204"
    V4_judge = 1
    V5 = "P206"
    V5_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge or df_data[V4].iloc[i] != V4_judge or df_data[V5].iloc[i] != V5_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                df_data.at[i,V4] = V4_judge
                df_data.at[i,V5] = V5_judge
                print("fixed...")
     ########



    #rule 8
    RULE = 8
    V1 = 'AIT201'
    V1_judge = 260
    V2 = "FIT201"
    V2_judge = 0.5
    V3 = "P201"
    V3_judge = 1
    V4 = "P202"
    V4_judge = 1


    flag = 0
    r1_idx = np.where(df_data[V1] > V1_judge)[0]
    r1_idx_and = np.where(df_data[V2].iloc[r1_idx] > V2_judge)[0]
    r1_idx_t = r1_idx_and + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V3].iloc[i] != V3_judge or df_data[V4].iloc[i] != V4_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V3] = V3_judge
                df_data.at[i,V4] = V4_judge
                print("fixed...")
     ########




    #rule 9
    RULE = 9
    V1 = 'AIT503'
    V1_judge = 260
    V2 = "FIT201"
    V2_judge = 0.5
    V3 = "P201"
    V3_judge = 1
    V4 = "P202"
    V4_judge = 1


    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_and = np.where(df_data[V2].iloc[r1_idx] > V2_judge)[0]
    r1_idx_t = r1_idx_and + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V3].iloc[i] != V3_judge or df_data[V4].iloc[i] != V4_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V3] = V3_judge
                df_data.at[i,V4] = V4_judge
                print("fixed...")
     ########




    #rule 10
    RULE = 10
    V1 = 'AIT202'
    V1_judge = 6.95
    V2 = "P203"
    V2_judge = 1
    V3 = "P204"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########




    #rule 11
    RULE = 11
    V1 = 'AIT203'
    V1_judge = 500
    V2 = "P205"
    V2_judge = 1
    V3 = "P206"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] > V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########




    #rule 12
    RULE = 12
    V1 = 'AIT203'
    V1_judge = 420
    V2 = "FIT201"
    V2_judge = 0.5
    V3 = "AIT402"
    V3_judge = 250
    V4 = "P205"
    V4_judge = 2
    V5 = "P206"
    V5_judge = 2

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_and = np.where(df_data[V2].iloc[r1_idx] > V2_judge)[0]
    r1_idx_and2 = np.where(df_data[V3].iloc[r1_idx] > V3_judge)[0]
    r1_idx_t = r1_idx_and2 + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V4].iloc[i] != V4_judge and df_data[V5].iloc[i] != V5_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V4] = V4_judge
#                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########





    #rule 13
    RULE = 13
    V1 = 'AIT402'
    V1_judge = 250
    V2 = "P205"
    V2_judge = 1
    V3 = "P206"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########






    #rule 14
    RULE = 14
    V1 = 'LIT301'
    V1_judge = 250
    V2 = "P301"
    V2_judge = 1
    V3 = "P302"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########





    #rule 15
    RULE = 15
    V1 = 'LIT401'
    V1_judge = 1000
    V2 = "P301"
    V2_judge = 1
    V3 = "P302"
    V3_judge = 1



    flag = 0
    r1_idx = np.where(df_data[V1] >= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########




    #rule 16
    RULE = 16
    V1 = 'LIT401'
    V1_judge = 250
    V2 = "P401"
    V2_judge = 1
    V3 = "P402"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########



    #rule 17
    RULE = 17
    V1 = 'LIT401'
    V1_judge = 250
    V2 = "UV401"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] <= V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
     ########



    #rule 18
    RULE = 18
    V1 = 'FIT401'
    V1_judge = 0.5
    V2 = "UV401"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
     ########



    #rule 19
    RULE = 19
    V1 = 'AIT402'
    V1_judge = 240
    V2 = "P403"
    V2_judge = 1
    V3 = "P404"
    V3_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx + df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge or df_data[V3].iloc[i] != V3_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                df_data.at[i,V3] = V3_judge
                print("fixed...")
     ########



    #rule 20
    RULE = 20
    V1 = 'UV401'
    V1_judge = 1
    V2 = "P501"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] == V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
     ########




    #rule 21
    RULE = 21
    V1 = 'FIT401'
    V1_judge = 0.5
    V2 = "P501"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] < V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")
     ########




    #rule 22
    RULE = 22
    V1 = 'LIT101'
    V1_judge = 1100
    V2 = "P601"
    V2_judge = 1

    flag = 0
    r1_idx = np.where(df_data[V1] > V1_judge)[0]
    r1_idx_t = r1_idx+df_rules["Time"].iloc[RULE]
    for i in r1_idx_t:
        if i<len(df_data):
            if df_data[V2].iloc[i] != V2_judge:
                y[i] = 1
                flag = 1
                df_data.at[i,V2] = V2_judge
                print("fixed...")
     ########


    return df_data

#y = RuleCheck(df_data)
#df_y = pd.DataFrame(y)
#df_y.to_csv('invariants check_Y.csv',index = False)
#
#plot(df_y,df_y_true)
#print('checking error...')
#f1 = f1_score(df_y_true,df_y,average='binary')
#precision = precision_score(df_y_true,df_y,average='binary')
#recall = recall_score(df_y_true,df_y,average='binary')
#print('Precision:',precision)
#print('recall:', recall)
#print('f1:',f1)
