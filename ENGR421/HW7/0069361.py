#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.spatial import distance
from scipy import stats
import pandas as pd


# In[2]:


images = np.genfromtxt("/Users/ozlemserifogullari/Documents/ENGR421/HW7/hw07_data_set_images.csv", delimiter = ",", skip_header = 0)
classes = np.genfromtxt("/Users/ozlemserifogullari/Documents/ENGR421/HW7/hw07_data_set_labels.csv", delimiter = ",", skip_header = 0, dtype=int)

train_indices = np.arange(0,2000)
test_indices = np.arange(2000,4000)
x_train = images[train_indices,:]
y_train = classes[train_indices]
x_test = images[test_indices,:]
y_test = classes[test_indices]

N = x_train.shape[0]
D = x_train.shape[1]
K = int(max(y_train))
N_test = x_test.shape[0]


# In[3]:


def calculate_total_wcs_mat(x, y, means):
    result = np.zeros((D,D))
    wcs_mat = [(np.dot(np.transpose(x[y == (c + 1)] - means[c]), (x[y == (c + 1)] - means[c]))) for c in range(K)]
    for i in range(D):
        for j in range(D):
            sum = 0
            for k in range(K):
                sum += wcs_mat[k][i][j]
            result[i][j] = sum
            
    return result

def calculate_bcs_mat(x, y, means):
    result = np.zeros((D,D))
    overall_mean = np.mean(x, axis = 0)
    Ni = [len(x[y == c+1]) for c in range(K)]
    for k in range(K):
        m = (means[k] - overall_mean).reshape(D,1)
        result += Ni[k] * np.dot(m, np.transpose(m))     
    return result


# In[4]:


class_means = np.array([np.mean(x_train[y_train == i + 1,] , axis=0) for i in range(K)])
SW = calculate_total_wcs_mat(x_train, y_train, class_means)
SB = calculate_bcs_mat(x_train, y_train, class_means)
print(SW[0:4, 0:4])
print(SB[0:4, 0:4])


# In[5]:


SW_inverse = np.linalg.inv(SW)
values, vectors = la.eig(np.dot(SW_inverse, SB))
values = np.real(values)
vectors = np.real(vectors)

print(values[0:9])

lnine_eigv = values[0:9]
eigvectors = vectors[:, 0:9]
#print(eigvectors)


# In[6]:


def calculate_z(x, mean, W_transpose):
    return np.dot(x - mean, W_transpose)


# In[7]:


# calculate two-dimensional projections
z_train = calculate_z(x_train, np.mean(x_train, axis = 0),vectors[:,[0, 1]])
z_test = calculate_z(x_test, np.mean(x_train, axis = 0),vectors[:,[0, 1]])
# plot two-dimensional projections
plt.figure(figsize = (10, 10))
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    plt.plot(z_train[y_train == c + 1, 0], z_train[y_train == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag","ankle boot" ],
           loc = "upper left", markerscale = 2)
plt.xlabel("#Component1")
plt.ylabel("#Component2")
plt.ylim(-6,6) # set the y limits
plt.xlim(-6,6) # set the x limits
plt.show()

plt.figure(figsize = (10, 10))
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    plt.plot(z_test[y_test == c + 1, 0], z_test[y_test == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag","ankle boot" ],
           loc = "upper left", markerscale = 2)
plt.xlabel("#Component1")
plt.ylabel("#Component2")
plt.ylim(-6,6) #set the y limits
plt.xlim(-6,6) #set the x limits
plt.show()


# In[8]:


# Learn a 𝑘-nearest neighbor classifier 
def k_nn(y, z, z_train, k):
    result = []
    for i in range(z.shape[0]):
        current = z[i,:]
        distances = np.zeros(z_train.shape[0])
        for j in range(z_train.shape[0]):
            distances[j] = distance.euclidean(current, z_train[j, :])
        nearests = np.argsort(distances)[:k]
        possible_y = []
        for i in nearests:
            possible_y.append(y[i])
        y_hat = stats.mode(possible_y)[0]
        result.append(y_hat)
    return result
            


# In[9]:


# calculate nine-dimensional projections
z_train = calculate_z(x_train, np.mean(x_train, axis = 0),vectors[:,0:9])
z_test = calculate_z(x_test, np.mean(x_train, axis = 0),vectors[:,0:9])

# set k = 11
y_pred_train = k_nn(y_train, z_train, z_train, 11)
y_pred_test = k_nn(y_train, z_test, z_train, 11)


# In[10]:


# confusion matrix
def print_confusion_matrix(y_truth, y_pred, no, colname):
    confusion_matrix = pd.crosstab(np.reshape(y_pred, no), y_truth,
                               rownames = ["y_hat"], colnames = [colname])
    print(confusion_matrix)


# In[11]:


print_confusion_matrix(y_train, y_pred_train, N, "y_train")
print_confusion_matrix(y_test, y_pred_test, N_test, "y_test")


# In[ ]:




