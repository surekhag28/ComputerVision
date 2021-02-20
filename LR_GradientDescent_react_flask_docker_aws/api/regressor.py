#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:53:02 2021

@author: surekhagaikwad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def compute_cost(x_val,y_val,m,c):
    n=len(x_val)
    cost=0
    for i in range(n):
        y_pred = m*x_val[i]+c
        cost+=(y_pred-y_val[i])**2
    return (cost/(2*n))


def train(file_path):
    data = pd.read_csv(file_path)
    
    x=data.iloc[:,0].values
    y=data.iloc[:,1].values
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=42)
    # Building the model
    m = 0
    c = 0
    m_hist=[]
    c_hist=[]
    cost_hist=[]
    
    L = 0.0001  # The learning Rate
    epochs = 1000  # The number of iterations to perform gradient descent
    
    n = float(len(x_train)) # Number of elements in X
    
    # Performing Gradient Descent 
    for i in range(epochs): 
        Y_pred = m*x_train + c  # The current predicted value of Y
        D_m = (-2/n) * sum(x_train * (y_train - Y_pred))  # Derivative wrt m
        D_c = (-2/n) * sum(y_train - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        m_hist.append(m)
        c_hist.append(c)
        cost_hist.append(compute_cost(x_train,y_train,m,c))
        
    rmse = rmse_evaluation(x_test,y_test,m,c)
    r2 = r_squared(x_test,y_test,m,c)
    
    plot_grad(x_train,y_train,m_hist,c_hist)
    plot_cost(m_hist,cost_hist)
    
    return rmse,r2,m,c
    

def plot_grad(x_train,y_train,m_hist,c_hist):
    plt.scatter(x_train,y_train,c='r')
    for i in range(len(m_hist)):
        y_pred = x_train*m_hist[i]+c_hist[i]
        plt.plot(x_train,y_pred)
    
    plt.xlabel('X-val')
    plt.ylabel('Y-val')
    plt.title('Gradient descent with iterations')
    plt.savefig('./uploads/grad.png')
    plt.close()

def plot_cost(m_hist,cost_hist):
    plt.plot(m_hist,cost_hist,c='green')
    plt.xlabel('m')
    plt.ylabel('Cost J(m)')
    plt.title('Cost vs Coefficient m')
    plt.savefig('./uploads/cost.png')
    plt.close()

# RMSE on test data

def rmse_evaluation(x_val,y_val,m,c):
    rmse=0
    n=len(y_val)
    for i in range(n):
        y_pred = m*x_val[i]+c
        rmse+=(y_val[i]-y_pred)**2
    return np.sqrt(rmse)

def r_squared(x_val,y_val,m,c):
    ssr=0
    sst=0
    y_mean=np.mean(y_val)
    n=len(x_val)
    for i in range(n):
        y_pred=m*x_val[i]+c
        ssr+=(y_val[i]-y_pred)**2
        sst+=(y_val[i]-y_mean)**2
    return 1-(ssr/sst)
