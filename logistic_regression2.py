from functions import *

def main():
    # Loading data
    cl_test2 = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=",")
    cl_train2 = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=",")

    n = cl_train2.shape[0]
    x0 = np.ones((n,1))
    x1 = cl_train2[:,0].reshape(n,1)
    x2 = cl_train2[:,1].reshape(n,1)
    x3 = (x1*x1).reshape(n,1)
    x4 = (x2*x2).reshape(n,1)

    X = np.hstack([x0,x1,x2,x3,x4])
    '''
    #Setting iteration number and learning rate
    numIterations = 10000
    learning_rate = 0.015

    # Formatting training data
    n = cl_train2.shape[0]
    m = cl_train2.shape[1]
    X = np.hstack([np.ones((n,1)), cl_train2[0:,:m-1]])
    y = cl_train2[0:,m-1:m]
    w = np.ones(m)
    w = w.reshape(m,1)

    # Running gradient descent
    w , e = gradient_descent(X, y, w, n, m, numIterations, learning_rate)

    # Formatting test
    n = cl_test2.shape[0]
    m = cl_test2.shape[1]
    X_test1 = np.hstack([np.ones((n,1)), cl_test2[0:,:m-1]])
    y_test1 = cl_test2[0:,m-1:m]

    # Predicting test data
    predict(w,X_test1,y_test1)

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


    # Plotting cross entropy error
    bx = fig.add_subplot(222, facecolor='#fff2f2')
    bx.set_title('Cross entropy error')
    bx.plot(range(1,numIterations + 1 ,1), e,  c='b')
    bx.axis([0, len(e), 0, 1])
    bx.set_xlabel('iterations')
    bx.set_ylabel('error')
    plt.tight_layout()
    plt.show()
    '''
main()
