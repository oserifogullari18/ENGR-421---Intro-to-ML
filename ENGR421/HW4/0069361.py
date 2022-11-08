#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statistics import mode


# In[ ]:


# read data into memory
data_set = np.genfromtxt("/Users/ozlemserifogullari/Documents/ENGR421/HW4/hw04_data_set_train.csv", delimiter = ",", skip_header = 0)

# get x and y values
x_train = data_set[:,0]
y_train = data_set[:,1]

# get number of samples
N = data_set.shape[0]


# In[ ]:


test_set = np.genfromtxt("/Users/ozlemserifogullari/Documents/ENGR421/HW4/hw04_data_set_test.csv", delimiter = ",", skip_header = 0)

# get x and y values
x_test = test_set[:,0]
y_test = test_set[:,1]

# get number of test case
N_test = test_set.shape[0]

    


# In[ ]:


#REGRESSOGRAM


def regressogram(x, y, lb, rb, no_bins):
    result = np.zeros(no_bins)
    
    for i in range(no_bins):
        result[i] = np.sum([(1 if x[n] > lb[i] and x[n] <= rb[i] else 0) * y[n] for n in range (N)]) / np.sum([1 if x[n] > lb[i] and x[n] <= rb[i] else 0 for n in range (N)])
                             
    return result

def plot_regressogram (regressogram, no_bin, lb, rb):
    plt.figure(figsize = (10, 5))
    plt.xlabel("Time(sec)")
    plt.ylabel("Signal(millivolt)")
    
    plt.plot(x_train, y_train, "b.", markersize = 10, label = "Training")
    plt.legend(loc = "upper right")
   
    for n in range(no_bin):
        plt.plot([lb[n], rb[n]], [regressogram[n], regressogram[n]], "k-")
    for n in range(no_bin - 1):
        plt.plot([rb[n], rb[n]], [regressogram[n], regressogram[n + 1]], "k-")
        
    plt.figure(figsize = (10, 5))
    plt.xlabel("Time(sec)")
    plt.ylabel("Signal(millivolt)")
    
    plt.plot(x_test, y_test, "r.", markersize = 10, label = "Test")
    plt.legend(loc = "upper right")
    
    for n in range(no_bin):
        plt.plot([lb[n], rb[n]], [regressogram[n], regressogram[n]], "k-")
    for n in range(no_bin - 1):
        plt.plot([rb[n], rb[n]], [regressogram[n], regressogram[n + 1]], "k-")


# In[ ]:


#parameters

bw = 0.1
origin = 0.0

minX = min(x_train)
maxX = max(x_train)
left_bound = np.arange(origin, maxX, bw)
right_bound = np.arange(origin + bw, maxX+ bw, bw)
no_bins = len(left_bound)


# In[ ]:


# calculate and plot regressogram
r = regressogram(x_train, y_train, left_bound, right_bound, no_bins)
plot_regressogram(r, no_bins, left_bound, right_bound)


# In[ ]:


def predict_reg(lb, rb, no_bin, test, regressogram):
    y_pred = np.zeros(N_test)
    for i in range (no_bin):
        for j in range (N_test):
            if test[j] > lb[i] and test[j] <= rb[i]:
                y_pred[j] = regressogram[i]
    return y_pred
                
def RMSE(y, y_head):
    sum = np.sum((y - y_head)**2)
    avg = float(sum) / N_test
    error = np.sqrt(avg)
    return error
    
def RMSE_reg(lb, rb, no_bin, regressogram) :
    sum = 0.0
    for i in range (no_bin):
        for j in range (N_test):
            if x_test[j] > lb[i] and x_test[j] <= rb[i]:
                sum = sum + (y_test[j] - regressogram[i])**2

    avg = sum / N_test
    error = np.sqrt(avg)
    return error


# In[ ]:


# predict values
y_pred_reg = predict_reg(left_bound, right_bound, no_bins, x_test, r)

# calculate RMSE for regressogram
print("Regressogram => RMSE is ", RMSE(y_test, y_pred_reg), " when h is ", bw)


# In[ ]:


# RUNNING MEAN SMOOTHER

def mean_smoother(x, y, interval, h):
    no_points = len(interval)
    result = np.zeros(no_points)
    
    for i in range (no_points):
        result[i] = np.sum([(1 if (interval[i] - 0.5*h) < x[n] and (interval[i] + 0.5*h) >= x[n] else 0) * y[n] for n in range (N)]) / np.sum([1 if (interval[i] - 0.5*h) < x[n] and (interval[i] + 0.5*h) >= x[n] else 0 for n in range (N)]) 
        
    return result

def predict_rms(rms, interval):
    y_pred = np.zeros(N_test)
    
    for i in range (N_test):
        # take the rms result of the data interval points that close the corresponding test value
        # without atol parameter the result is always an empty set so I added an arbitrary number
        possible = rms[np.where(np.isclose(interval, x_test[i], 0.0008))]
        if len(possible) != 0: #if there is any close data interval points predict as the first result
            y_pred[i] = possible[0]
        elif i!= 0: # otherwise if i is not 0 predict as the average of previous predictions
            y_pred[i] = np.average(y_pred[0:i])
        
    return y_pred

def plot_rms(interval, rms):
    # training data
    plt.figure(figsize = (10, 5))
    plt.xlabel("Time(sec)")
    plt.ylabel("Signal(millivolt)")
    
    plt.plot(x_train, y_train, "b.", markersize = 10, label = "Training")
    plt.legend(loc = "upper right")
    plt.plot(data_interval, rms, "k-")
    
    # test data
    plt.figure(figsize = (10, 5))
    plt.xlabel("Time(sec)")
    plt.ylabel("Signal(millivolt)")
    
    plt.plot(x_test, y_test, "r.", markersize = 10, label = "Test")
    plt.legend(loc = "upper right")
    plt.plot(data_interval, rms, "k-")


# In[ ]:


# parameters

bw = 0.1


# In[ ]:


data_interval = np.linspace(minX, maxX, 1601)
rms = mean_smoother(x_train, y_train, data_interval, bw)
plot_rms(data_interval, rms)


# In[ ]:


# predict values
y_pred_rms = predict_rms(rms, data_interval)

# calculate RMSE for running mean smoother
print("Running Mean Smoother  => RMSE is ", RMSE(y_test, y_pred_rms), " when h is ", bw)


# In[ ]:


# KERNEL SMOOTHER

def kernel_smoother(x, y, interval, h):
    no_points = len(interval)
    result = np.zeros(no_points)
    for i in range (no_points):
        result[i] = np.sum([(1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (interval[i] - x[n])**2 / h**2)) * y[n] for n in range(N)]) / np.sum([1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (interval[i] - x[n])**2 / h**2) for n in range(N)])
    
    return result

def predict_ks(ks, interval):
    y_pred = np.zeros(N_test)
    
    for i in range (N_test):
        # take the rms result of the data interval points that close the corresponding test value
        # without atol parameter the result is always an empty set so I added an arbitrary number
        possible = ks[np.where(np.isclose(interval, x_test[i], 0.0008))]
        if len(possible) != 0: # if there is any close data interval points predict as the first result 
            y_pred[i] = possible[0]
        elif i != 0: # otherwise if i is not 0 predict as the average of previous predictions
            y_pred[i] = np.average(y_pred[0:i])
        
    return y_pred

def plot_ks(interval, ks):
    # training data
    plt.figure(figsize = (10, 5))
    plt.xlabel("Time(sec)")
    plt.ylabel("Signal(millivolt)")
    
    plt.plot(x_train, y_train, "b.", markersize = 10, label = "Training")
    plt.legend(loc = "upper right")
    plt.plot(data_interval, ks, "k-")
    
    # test data
    plt.figure(figsize = (10, 5))
    plt.xlabel("Time(sec)")
    plt.ylabel("Signal(millivolt)")
    
    plt.plot(x_test, y_test, "r.", markersize = 10, label = "Test")
    plt.legend(loc = "upper right")
    plt.plot(data_interval, ks, "k-")


# In[ ]:


# parameters

bw = 0.02


# In[ ]:


data_interval = np.linspace(minX, maxX, 1601)
ks = kernel_smoother(x_train, y_train, data_interval, bw)
plot_rms(data_interval, ks)


# In[ ]:


# predict values
y_pred_ks = predict_ks(ks, data_interval)

# calculate RMSE for running mean smoother
print("Kernel Smoother  => RMSE is ", RMSE(y_test, y_pred_ks), " when h is ", bw)


# In[ ]:




