from functions import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Loading data
    cl_test2 = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=",")
    cl_train2 = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=",")

    # Formatting training data
    n,m = cl_train2.shape
    y = cl_train2[0:,m-1:m]
    x0 = np.ones((n,1))
    x1 = cl_train2[:,0].reshape(n,1)
    x2 = cl_train2[:,1].reshape(n,1)
    x3 = (x1*x1).reshape(n,1)
    x4 = (x2*x2).reshape(n,1)
    X = np.hstack([x0,x1,x2,x3,x4])
    m = X.shape[1]
    w = np.ones(m).reshape(m,1)

    #Setting iteration number and learning rate
    numIterations = 10000
    learning_rate = 0.01

    # Running gradient descent
    w , e = gradient_descent(X, y, w, n, m, numIterations, learning_rate)

    # Formatting test
    n,m = cl_test2.shape
    y_test = cl_test2[0:,m-1:m]
    x0_test = np.ones((n,1))
    x1_test = cl_test2[:,0].reshape(n,1)
    x2_test = cl_test2[:,1].reshape(n,1)
    x3_test = (x1_test*x1_test).reshape(n,1)
    x4_test = (x2_test*x2_test).reshape(n,1)
    X_test = np.hstack([x0,x1,x2,x3,x4])
    m = X_test.shape[1]

    # Predicting test data
    print(w)
    predict(w,X_test,y_test)


    # Plotting training data
    fig = plt.figure()
    ax = fig.add_subplot(221)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Decision boundary')
    scatter_plot(cl_train2,ax,shape='.',positive_color='#40cc49', negative_color='#ff6666',dotsize=5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Plotting test boundary
    ax = fig.add_subplot(221)
    scatter_plot(cl_test2,ax,shape='v',positive_color='g', negative_color='r',dotsize=2)

    # Plotting decision boundary
    w0,w1,w2,w3,w4 = w.T
    x1 = np.arange(0., 1, .1)
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x,y)
    F = w0 + w1*X + w2*Y + w3*(X**2) + w4*(Y**2)
    plt.contour(X,Y,F,[0])
    ax.axis([0, 1, 0, 1])

    # Plotting cross entropy error
    bx = fig.add_subplot(222)
    bx.set_title('Cross entropy error')
    bx.plot(range(1,numIterations + 1 ,1), e,  c='b')
    bx.axis([0, len(e), 0, 1])
    bx.set_xlabel('iterations')
    bx.set_ylabel('error')
    plt.tight_layout()
    plt.show()

main()
