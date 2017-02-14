from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(data,ax,shape,positive_color,negative_color,dotsize):
    # Dot area/size
    area = np.pi * (dotsize**2)
    rows, colms = data.shape
    for i in range(rows):
        if data[i][colms-1] == 1:
            ax.scatter(data[i][0], data[i][1],s = area, c=positive_color,alpha=0.5,marker=shape)
        else:
            ax.scatter(data[i][0], data[i][1],s = area, c=negative_color,alpha=0.5,marker=shape)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(loss,X):
    cost = X*loss
    cost_sum = cost.sum(0)
    return cost_sum.reshape(cost_sum.shape[0],1)

def gradient_descent(X, y, w, n, m, numIterations, learning_rate):
    error_ce = []
    for i in range(numIterations):
        sigma = sigmoid(np.dot(X, w))
        loss = sigma - y
        error_ce.append(cross_entropy_error(n,y,sigma))
        cost = compute_cost(loss,X)
        w = w - learning_rate*cost
    return (w.T,error_ce)

# Does not work with 5 weights
def predict(w,X,y):
    z = np.dot(X,w.T)
    sigma = sigmoid(z)
    predict = np.array(list(map(lambda x : 1 if x>0.5 else 0, sigma )))
    n = y.shape[0]
    score = 0
    for i in range(n):
        if y[i][0] == predict[i]:
            score += 1
    print(y)
    print(predict)
    print("scored: "+ str((score/n)*100)+"%")

def cross_entropy_error(n,y,sigma):
    return(-(1/n)*np.sum(y*np.log(sigma) + (1-y)*np.log(1-sigma)))
