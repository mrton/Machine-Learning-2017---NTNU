import numpy as np
import random


cl_test1 = np.genfromtxt('./datasets/classification/cl-test-1.csv', delimiter=",")
cl_test2 = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=",")
cl_train1 = np.genfromtxt('./datasets/classification/cl-train-1.csv', delimiter=",")
cl_train2 = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=",")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(loss,X):
    numExamples = X.shape[0]
    for i in range(numExamples):
        X[0] = X[0]*loss[0]
    cost = X.sum(axis=0)
    return cost.reshape(cost.shape[0],1)

# num of examples
N = cl_train1.shape[0]
# num features
m = cl_train1.shape[1]

X = np.hstack([np.ones((N,1)), cl_train1[0:,:2]])
y = cl_train1[0:,2:3]
w = np.ones(m)
w = w.reshape(3,1)
alpha = 0.001
numIterations = 1000

sigma = sigmoid(np.dot(X, w))
loss = sigma - y
cost = (compute_cost(loss,X))
w = w - alpha*cost
print(w)





def gradient_descent(X, y, theta, n, numIterations):
    x_transpose = X.transpose()
    h = np.dot(x)

'''
def update(w,learning_rate,X,y,iterations):
    count = 0
    while count < iterations:

    w_next = w - np.sum(learning_rate)
    learning_rate*(sigmoid( w.transpose()* ))
    count += 1
'''
