# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 01:41:35 2019

@author: jiayi
"""

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


df_train = pd.read_csv("WADI_attackdata.csv")
df_head = pd.read_csv("WADI_normal.csv",nrows = 1)
header = df_head.columns
df_train.drop('Row',axis=1, inplace=True)
df_train.drop('Date',axis=1, inplace=True)
df_train.drop('Time',axis=1, inplace=True)

df = pd.DataFrame(np.array(df_train),columns = header)

df.to_csv("WADI_attack.csv",index = False)

# 28/12/2015 10:00:00 AM


#####################
df_time = df_train[["Date","Time"]]
df_time.to_csv("WADI_time.csv",index = False)
