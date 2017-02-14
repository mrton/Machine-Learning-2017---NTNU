from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def main():



    cl_test1 = np.genfromtxt('./datasets/classification/cl-test-1.csv', delimiter=",")
    cl_test2 = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=",")
    cl_train1 = np.genfromtxt('./datasets/classification/cl-train-1.csv', delimiter=",")
    cl_train2 = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=",")

    numIterations = 1000
    learning_rate = 0.01

    n = cl_train1.shape[0]
    m = cl_train1.shape[1]
    X = np.hstack([np.ones((n,1)), cl_train1[0:,:m-1]])
    y = cl_train1[0:,m-1:m]
    w = np.ones(m)
    w = w.reshape(m,1)

    # Running gradient descent
    w , e = gradient_descent(X, y, w, n, m, numIterations, learning_rate)

    n = cl_test1.shape[0]
    m = cl_test1.shape[1]
    X_test1 = np.hstack([np.ones((n,1)), cl_test1[0:,:m-1]])
    y_test1 = cl_test1[0:,m-1:m]
    predict(w,X_test1,y_test1)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    scatter_plot(cl_train1,ax)
    bx = fig.add_subplot(222, facecolor='#fff2f2') # creates 2nd subplot with yellow background
    bx.plot(range(1,numIterations + 1 ,1), e,  c='b')
    bx.axis([0, len(e), 0, 1])
    plt.show()

def scatter_plot(data,ax):
    #dot area/size
    area = np.pi * (4**2)
    rows, colms = data.shape
    for i in range(rows):
        if data[i][colms-1] == 1:
            ax.scatter(data[i][0], data[i][1],s = area, c='#40cc49')
        else:
            ax.scatter(data[i][0], data[i][1],s = area, c='#ff6666')

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

def predict(w,X,y):
    z = np.dot(X,w.T)
    sigma = sigmoid(z)
    predict = np.array(list(map(lambda x : 1 if x>0.5 else 0, sigma )))
    n = y.shape[0]
    score = 0
    for i in range(n):
        if y[i][0] == predict[i]:
            score += 1
    print("scored: "+ str((score/n)*100)+"%")


def cross_entropy_error(n,y,sigma):
    return(-(1/n)*np.sum(y*np.log(sigma) + (1-y)*np.log(1-sigma)))

main()
