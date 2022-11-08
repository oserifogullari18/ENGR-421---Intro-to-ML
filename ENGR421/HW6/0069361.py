#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt
import math


# In[2]:


images = np.genfromtxt("/Users/ozlemserifogullari/Documents/ENGR421/HW6/hw06_data_set_images.csv", delimiter = ",", skip_header = 0)
classes = np.genfromtxt("/Users/ozlemserifogullari/Documents/ENGR421/HW6/hw06_data_set_labels.csv", delimiter = ",", skip_header = 0)

train_indices = np.arange(0,1000)
test_indices = np.arange(1000,2000)
x_train = images[train_indices,:]
y_train = classes[train_indices]
x_test = images[test_indices,:]
y_test = classes[test_indices]

N = len(x_train)
D = len(x_train[0])
K = 2 # 1 and -1

N_test = len(x_test)

color_size = 256 # the maximum value of a pixel can have


# In[3]:


def calculate_hist(x, no_bins, no_pixels, no):
    result = np.zeros((no, no_bins))
    bin_width = 256 / no_bins
    left_borders = np.arange(0, 256, bin_width)
    right_borders = np.arange(0 + bin_width, 256 + bin_width , bin_width)
    for i in range(no_bins):
        for j in range(no):
            result[j][i] = np.sum([1 if x[j][d] >= left_borders[i] and x[j][d] < right_borders[i] else 0 for d in range(no_pixels)]) / no_pixels
    
    return result

L = 64
H_train = calculate_hist(x_train, L, D, N)
H_test = calculate_hist(x_test, L, D, N_test)

print(H_train[0:5, 0:5])
print(H_test[0:5, 0:5])  


# In[4]:


def hist_intersection(h1, h2, l):
    K = np.zeros((len(h1), len(h2)))
    for r1 in range (len(h1)):
            for r2 in range (len(h2)):
                sum = 0
                for bin in range(l):
                    sum += min(h1[r1][bin], h2[r2][bin])
                K[r1][r2] = sum
    return K
                            
K_train = hist_intersection(H_train, H_train, L)
K_test = hist_intersection(H_test, H_train, L)

print(K_train[0:5, 0:5])
print(K_test[0:5, 0:5]) 


# In[5]:


def qp_problem(K, C, no, y, epsilon):
    yyK = np.matmul(y_train[:,None], y_train[None,:]) * K

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((no, 1)))
    G = cvx.matrix(np.vstack((-np.eye(no), np.eye(no))))
    h = cvx.matrix(np.vstack((np.zeros((no, 1)), C * np.ones((no, 1)))))
    A = cvx.matrix(1.0 * y[None,:])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    return alpha, w0

def svm_predict(K, y, alpha, w0):
    # calculate predictions on training samples
    f_predicted = np.matmul(K, y[:,None] * alpha[:,None]) + w0

    # calculate confusion matrix
    y_predicted = 2 * (f_predicted > 0.0) - 1
    
    return y_predicted

def print_confusion_matrix(y_truth, y_pred, no):
    confusion_matrix = pd.crosstab(np.reshape(y_pred, no), y_truth,
                               rownames = ["y_predicted"], colnames = ["y_train"])
    print(confusion_matrix)


# In[6]:


alpha, w0 = qp_problem(K_train, 10, N, y_train, 0.001)

y_pred_train = svm_predict(K_train, y_train, alpha, w0)
print_confusion_matrix(y_train, y_pred_train, N)

y_pred_test = svm_predict(K_test, y_train, alpha, w0)
print_confusion_matrix(y_test, y_pred_test, N_test)


# In[7]:


def calculate_accuracy(y_truth, y_pred, no):
    accuracy = 0
    no = y_truth.shape[0]
    
    for i in range(no):
        if y_truth[i] == y_pred[i]:
            accuracy += 1
    
    return accuracy / no

c_array = np.array([pow(10,-1), pow(10, -0.5), pow(10, 0), pow(10, 0.5), pow(10, 1), pow(10, 1.5), pow(10, 2), pow(10, 2.5), pow(10, 3)])
train_acc = []
test_acc = []

for c in c_array:
    alpha, w0 = qp_problem(K_train, c, N, y_train, 0.001)
    y_pred_train = svm_predict(K_train, y_train, alpha, w0)
    acc = calculate_accuracy(y_train, y_pred_train, N)
    train_acc.append(acc)
    
    y_pred_test = svm_predict(K_test, y_train, alpha, w0)
    acc = calculate_accuracy(y_test, y_pred_test, N_test)
    test_acc.append(acc)


# In[8]:


def plot_graph(c_values, train_accuracy, test_accuracy):
    from matplotlib import ticker # to format the graph
    
    f = plt.figure(figsize = (12, 5))
    f_x = f.add_subplot(1,1,1)
    plt.xlabel("Regularization parameter (C)")
    plt.ylabel("Accuracy")
    plt.scatter(c_values,train_accuracy)
    plt.plot(c_values,train_accuracy,label = "training", color = "blue")
    plt.scatter(c_values,test_accuracy)
    plt.plot(c_values,test_accuracy,label = "test", color = "red")
    f_x.set_xscale('symlog')
    f_x.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    plt.legend(loc = "upper left")
    
    
plot_graph(c_array, train_acc, test_acc)


# In[ ]:




