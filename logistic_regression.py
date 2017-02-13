import numpy as np
import matplotlib.pyplot as plt

cl_test1 = np.genfromtxt('./datasets/classification/cl-test-1.csv', delimiter=",")
cl_test2 = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=",")
cl_train1 = np.genfromtxt('./datasets/classification/cl-train-1.csv', delimiter=",")
cl_train2 = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=",")
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(loss,X):
    numExamples = X.shape[0]
    for i in range(numExamples):
        X[i] = X[i]*loss[i]
    cost = X.sum(axis=0)
    return cost.reshape(cost.shape[0],1)

def gradient_descent(X, y, w, n, m, numIterations, learning_rate):
    for i in range(numIterations):
        sigma = sigmoid(np.dot(X, w))
        loss = sigma - y
        cost = (compute_cost(loss,X))
        w = w - learning_rate*cost
        #print(w.transpose())
    print(w.transpose())

# PARAMETERS
# num of examples
n = cl_train1.shape[0]
# num features
m = cl_train1.shape[1]
X = np.hstack([np.ones((n,1)), cl_train1[0:,:m-1]])
y = cl_train1[0:,m-1:m]
w = np.ones(m)
w = w.reshape(m,1)

gradient_descent(X, y, w, n, m, numIterations=10000, learning_rate=0.01)

# For plotting
positive = []
negative = []
for example in cl_train1:
    if example[m-1] == 1:
        positive.append(example)
    else:
        negative.append(example)
X_pos = []
Y_pos = []
X_neg = []
Y_neg = []
for i in positive:
    X_pos.append(i[0])
    Y_pos.append(i[1])
for i in negative:
    X_neg.append(i[0])
    Y_neg.append(i[1])

plt.plot(X_pos, Y_pos, 'b.')
plt.plot(X_neg, Y_neg, 'r.')
plt.axis([0, 1, 0, 1])
plt.show()
