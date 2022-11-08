from ctypes import sizeof
import math
from cv2 import mean
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import pandas as pd

# read csv file into memory based lab1
images = np.genfromtxt("/Users/ozlemserifogullari/Documents/ENGR421/HW2/hw02_data_set_images.csv", delimiter = ",", skip_header = 0)

#creating data set
points1 = [images[i] for i in range(25)]
points2 = [images[i] for i in range(39,64)]
points3 = [images[i] for i in range(78,103)]
points4 = [images[i] for i in range(117, 117+25)]
points5 = [images[i] for i in range(156, 156+25)]


x = np.concatenate((points1, points2, points3,points4,points5))

#since the data lables are ordered (explained in homework description) it can be implemented as below
y = np.concatenate((np.repeat(1, 25), np.repeat(2, 25), np.repeat(3, 25), np.repeat(4, 25), np.repeat(5, 25)))
y_test = np.concatenate((np.repeat(1, 14), np.repeat(2, 14), np.repeat(3, 14), np.repeat(4, 14), np.repeat(5, 14)))

N = len(x) #number of data points
K = np.max(y) #number of classes
D = x[0].shape #dimension of data vectors

#creating test cases
test_data = np.concatenate((np.copy(images[25:39]), np.copy(images[64:78]), np.copy(images[103:117]), np.copy(images[117+25:156]), np.copy(images[156+25:195])))

#estimating pcds
# calculate sample means based lab1
mean_vectors = [np.mean(x[y == (c + 1)], axis = 0) for c in range(K)]

#calculate sample varience matrices
cov_matricies = [(np.matmul(np.transpose(x[y == (c + 1)] - mean_vectors[c]), (x[y == (c + 1)] - mean_vectors[c])) / 25) for c
    in range(K)]

"THE COVARIENCE MATRIX I GOT IS SINGULAR I COUND'T FIX IT BUT THE REST SHOULD BE IMPLEMENTED AS BELOW"

#calculate pcds
#distribution for one class c
def distribution_c (x,mean,cov):
     coef = 1 / (math.sqrt(2*math.pi*np.linalg.det(cov)))
     mult = coef * np.exp(-0.5*np.matmul(np.matmul(np.transpose(x-mean), np.linalg.inv(cov)),(x-mean)))
     return mult

pcd = [distribution_c(x[y == c+1], mean_vectors[c], cov_matricies[c]) for c in range(K)]
#estimate class priors
class_priors = [np.sum(y == c + 1) / N for c in range(K)]
print(class_priors)


#5-)Calculate confusion matrix with train data

#calculate score functions
def score_c(mean, cov, prior, x_test):
    W = -0.5*np.linalg.inv(cov)
    w = np.matmul(np.linalg.inv(cov), mean)
    w0 = (-0.5 * np.matmul(np.transpose(mean), np.linalg.inv(cov), mean)) + (-0.5*D*math.log(math.pi)) + (-0.5*math.log(np.linalg.det(cov))) + math.log(prior)
    return np.matmul(np.matmul(np.transpose(x_test), W), x_test) + np.matmul(np.transpose(w), x_test) + w0

y_predicted = []
for data_vector in x:
    score_funcs = [score_c(mean_vectors[c], cov_matricies[c], class_priors[c], data_vector) for c in range(K)]
    y_predicted.append(np.argmax(score_funcs) +1)

confusion_matrix = pd.crosstab(y_predicted, y_test, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


#6-)Calculate confusion matrix with test data

y_predicted = []
for x_test in test_data:
    score_funcs = [score_c(mean_vectors[c], cov_matricies[c], class_priors[c], x_test) for c in range(K)]
    y_predicted.append(np.argmax(score_funcs) +1)

confusion_matrix = pd.crosstab(y_predicted, y_test, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


