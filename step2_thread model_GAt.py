# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:29:30 2019

@author: yifan
"""

import numpy as np
import random
#import tensorflow as tf
import pandas as pd
#import time
import copy
#from keras import backend as K
import matplotlib.pyplot as plt
#from keras.models import load_model
#import keras
#from LSTM_for_SWaT import predict, plot, check
#from keras.optimizers import Adam
import math
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import preprocessing
from step2_rules_noprint import RuleCheck_all, RuleCheck_all_fix
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance


STATUS = "ALL"
MODEL = 'SWaT.hdf5'
ATTACKfile = "attack_x.csv"
ADVfile = "X_ADV_ALL10.csv"
ADV_rule = "X_ADV_ALL10_rule.csv"
header_sen = pd.read_csv("attack_x_sensor.csv").columns
header_act = pd.read_csv("attack_x_actuator.csv").columns
header = pd.read_csv(ATTACKfile).columns

actuator_head_MV = []
actuator_head_P = []
for i in header_act:
    if "MV" in i:
        actuator_head_MV.append(i)
    else:
        actuator_head_P.append(i)
WINDOW= 12

POP_SIZE = 100                      # population size
CROSS_RATE = 0.4                    # mating probability (DNA crossover)
MUTATION_RATE = 0.01                # mutation probability
N_GENERATIONS = 1000
DNA_SIZE_MV = 6#(12,51)#INPUT.shape(1)
DNA_SIZE_P = 19
DNA_SIZE_sen = 26
BOUND_MV = [0,0.5,1]
BOUND_P = [0,1]
BOUND_sen = (0,100)#remenber to divided by 100 to get decimal




df_X = pd.read_csv(ADVfile)
length = int(len(df_X)/WINDOW)
df_X = np.array(df_X).reshape(length,WINDOW,len(header))
df_check = pd.read_csv(ADVfile)
df_check = np.array(df_X).reshape(length,WINDOW,len(header))







def PreProcess(STATUS):
    NORMALfile = "normal_all.csv"

    df_train = pd.read_csv(NORMALfile)
    df_tr = df_train[4000:]

    #Data standardization
    scaler = preprocessing.StandardScaler().fit(df_tr)
    data_stand = scaler.transform(df_tr)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(data_stand)


    return scaler,min_max_scaler

scaler,min_max_scaler = PreProcess(STATUS)



class GA(object):
    def __init__(self, DNA_size_mv,DNA_size_p, DNA_bound_mv,DNA_bound_p, cross_rate, mutation_rate, pop_size,i):
        self.DNA_size_mv = DNA_size_mv
        self.DNA_size_p = DNA_size_p
        #DNA_bound[1] += 1
        self.DNA_bound_mv = DNA_bound_mv
        self.DNA_bound_p = DNA_bound_p
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.i = i

        population = np.concatenate((np.random.choice(self.DNA_bound_mv, size=(self.pop_size, self.DNA_size_mv)),(np.random.choice(self.DNA_bound_p, size=(self.pop_size, self.DNA_size_p)))),axis = 1)
        popA = pd.DataFrame(population,columns = actuator_head_MV +actuator_head_P)
        pop = popA[header_act]

        self.pop = np.array(pop)



    def get_fitness(self,df_adv,check_data_x,i):                      #calculate a fitness number
        check_data_org = scaler.inverse_transform(min_max_scaler.inverse_transform(check_data_x))
        check_data_org = pd.DataFrame(check_data_org,columns = header).round(2)
        y_original = RuleCheck_all(check_data_org,header)
        fitness = []
        for item in self.pop:
            df_adv.at[i,header_act] = item
            check_data_adv = scaler.inverse_transform(min_max_scaler.inverse_transform(df_adv))
            check_data_adv = pd.DataFrame(check_data_adv,columns = header).round(2)
            y_check = RuleCheck_all(check_data_adv,header)
            flag = 1
            if y_original[i]==0 and y_check[i]!=0:
                flag = 0
            fitness.append(flag)

                    #print("thread will be found by rules")
    #            if y_original[ii]==1 and y_check[ii]!=1:
    #                flag = 2
        return np.array(fitness)

    def check_rule(self,df_adv,check_data_x):
        check_data_adv = scaler.inverse_transform(min_max_scaler.inverse_transform(df_adv))
        check_data_adv = pd.DataFrame(check_data_adv,columns = header).round(2)
        check_data_org = scaler.inverse_transform(min_max_scaler.inverse_transform(check_data_x))
        check_data_org = pd.DataFrame(check_data_org,columns = header).round(2)
        y_original = RuleCheck_all(check_data_org,header)
        y_check = RuleCheck_all(check_data_adv,header)
        list_rule_fail = []
        for ii in range(len(y_original)):
            if y_original[ii]==0 and y_check[ii]!=0:
                list_rule_fail.append(ii)
        return list_rule_fail


    def select(self):
        fitness = self.get_fitness(df_adv,check_data_x,i) + 1e-4     # add a small amount to avoid all zero fitness
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.choice(self.DNA_bound_p)  # choose a random index
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

if __name__ == '__main__':
    data_adv_rule = []
    num_change = []
    for num in range(length):
        df_adv = pd.DataFrame(df_X[num], columns = header)
        check_data_x = pd.DataFrame(df_check[num], columns = header)

        ga = GA(DNA_size_mv = DNA_SIZE_MV,DNA_size_p = DNA_SIZE_P, DNA_bound_mv=BOUND_MV,DNA_bound_p=BOUND_P, cross_rate=CROSS_RATE,mutation_rate=MUTATION_RATE, pop_size=POP_SIZE,i = 0)
        list_rule_fail = ga.check_rule(df_adv,check_data_x)
        for i in list_rule_fail:
            num_change.append(i)
            print("array with failed points::::::",list_rule_fail)
            for generation in range(N_GENERATIONS):
                fitness = ga.get_fitness(df_adv,check_data_x,i)
                best_DNA = ga.pop[np.argmax(fitness)]
                print('Gen', generation, ': ', best_DNA)
                df_adv.at[i,header_act] = best_DNA
                print(best_DNA[3])
                list_rule_fail = ga.check_rule(df_adv,check_data_x)
                if i not in list_rule_fail:
                    print("solve!!!!!!")
                    break
                ga.evolve()
        data_adv_rule.append(df_adv)


    reshape_adv = np.reshape(np.array(data_adv_rule),(len(data_adv_rule)*WINDOW,len(header)))
    df_adv = pd.DataFrame(reshape_adv,columns = header)
    #Write and read
    df_adv.to_csv(ADV_rule,index = False)













