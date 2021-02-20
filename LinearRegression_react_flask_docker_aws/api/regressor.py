#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:25:37 2021

@author: surekhagaikwad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.Logger('Hello World')

# computing coefficients of best fit line
def coefficients(x_val,y_val):
    b0=0
    b1=0
    x_mean = np.mean(x_val)
    y_mean = np.mean(y_val)
    
    n = len(x_val)
    cova=0
    var=0
    for i in range(n):
        cova+= (x_val[i]-x_mean)*(y_val[i]-y_mean)
        var+= (x_val[i]-x_mean)**2
    b1 = cova/var
    b0 = y_mean - (b1*x_mean)
    
    return b1,b0

def train(file_path):
    data = pd.read_csv(file_path)

    x=data['Head Size(cm^3)'].values
    y=data['Brain Weight(grams)'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    b1,b0=coefficients(x_train,y_train)
    rmse=rmse_evaluation(x_test,y_test,b0,b1)
    r2=r_squared(x_test,y_test,b0,b1)
    
    
    # Plotting the best fit line
    x_max = np.max(x_train)+100
    x_min = np.min(x_train)-100
    
    x = np.linspace(x_min,x_max,1000)
    y = b0 + b1*x
    
    plt.plot(x,y,c='g',label='Linear regression line')
    plt.scatter(x_train,y_train,c='r',label='Actual data points')
    plt.xlabel('Head Size (cm^3)')
    plt.ylabel('Brain Weight (grams)')
    plt.legend(loc="upper left")
    plt.savefig('./uploads/plot.jpeg')
    
    return rmse,r2,b0,b1
    
def plot():
    return 'uploads/plot.jpeg'

def rmse_evaluation(x_val,y_val,b0,b1):
    n = len(x_val)
    rmse=0
    for i in range(n):
        y_pred = b0+(b1*x_val[i])
        rmse+= (y_val[i]-y_pred)**2
    return np.sqrt(rmse/n)

def r_squared(x_val,y_val,b0,b1):
    ssr=0
    sst=0
    n = len(x_val)
    y_mean = np.mean(y_val)
    for i in range(n):
        y_pred = b0+b1*x_val[i]
        ssr+= (y_val[i]-y_pred)**2
        sst+= (y_val[i]-y_mean)**2
    return 1 - (ssr/sst)