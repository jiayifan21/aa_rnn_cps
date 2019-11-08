# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:52:17 2019

@author: jiayi
"""

import pandas as pd
import datetime
import numpy as np

def timeChange(column,Format,Format2):
    print('111111111111111111')
    list_time = []
    for i in column:
        if str(i)[0] != ' ':
            k = pd.to_datetime(i,format=Format)#.value //10**6
            list_time.append(k)
        else:
            k = pd.to_datetime(i,format=Format2)#.value //10**6
            list_time.append(k)
    return list_time
    
    


df_exl = pd.read_excel('WADI_table.xlsx')
df_data = pd.read_csv('WADI_time.csv')

df_data["Timestamp"] = df_data["Date"].map(str) +" " +df_data["Time"]

9/10/17 19:25:00


list_StartTime = timeChange(df_exl['Starting Time'],'%d/%m/%Y %H:%M:%S','%Y-%m-%d %H:%M:%S')
list_EndTime = timeChange(df_exl['Ending Time'],'%d/%m/%Y %H:%M:%S','%Y-%m-%d %H:%M:%S')
list_time = timeChange(df_data['Timestamp'],'%m/%d/%Y %I:%M:%S.%f %p',' %d/%m/%Y %I:%M:%S %p')
df_exl['start'] = list_StartTime
df_exl['end'] = list_EndTime
df_data['Timestamp'] = list_time


df_data['Y'] = [0]*len(df_data)



for i in range(len(df_exl)):        
    start_time = df_exl['start'].iloc[i]
    end_time = df_exl['end'].iloc[i]
    list_index = df_data[df_data['Timestamp'].between(start_time,end_time)].index.tolist()
    print(start_time,end_time)
    print(len(list_index))
#    df_data['act_change'][list_index] = df_exl['actual_change'].iloc[i]
#    df_data['sen_act'][list_index] = df_exl['sen_act'].iloc[i]
    df_data['Y'][list_index] = 1

df_data['Y'].to_csv("WADI_attack_Y.csv")

#Prepare data   

df_actChan = df_data['act_change'] 
df_senAct = df_data['sen_act']

df_data = df_data.drop('act_change',axis =1)
df_data = df_data.drop('sen_act',axis =1)
df_data = df_data.drop('Timestamp',axis =1)
df_data = df_data.drop('Normal/Attack',axis =1)

df_data = df_data.fillna(0)
df_data.to_csv("attack_x.csv",index = False)


df_a = pd.DataFrame()
df_s = pd.DataFrame()
df_stg1 = pd.DataFrame()

df_a['act_change'] = df_actChan
df_s['sen_act'] = df_senAct
df_stg1['stage1'] = df_data['stage1']
df_stg1 = df_stg1.fillna(0)

df_a.to_csv("attack_y_chan.csv",index = False)
df_s.to_csv("attack_y_sen.csv",index = False)

df_stg1.to_csv("attack_y_stage1.csv",index = False)


