from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

cl_test1 = np.genfromtxt('./datasets/classification/cl-test-1.csv', delimiter=",")
cl_test2 = np.genfromtxt('./datasets/classification/cl-test-2.csv', delimiter=",")
cl_train1 = np.genfromtxt('./datasets/classification/cl-train-1.csv', delimiter=",")
cl_train2 = np.genfromtxt('./datasets/classification/cl-train-2.csv', delimiter=",")

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


n = cl_train1.shape[0]
m = cl_train1.shape[1]
X = np.hstack([np.ones((n,1)), cl_train1[0:,:m-1]])
y = cl_train1[0:,m-1:m]
w = np.ones(m)
w = w.reshape(m,1)
w , e = gradient_descent(X, y, w, n, m, numIterations=1000, learning_rate=0.01)


n = cl_test1.shape[0]
m = cl_test1.shape[1]
X_test1 = np.hstack([np.ones((n,1)), cl_test1[0:,:m-1]])
y_test1 = cl_test1[0:,m-1:m]
predict(w,X_test1,y_test1)

'''
# For plotting
positive = []
negative = []
for example in cl_train1:
    if example[m-1] == 1:
        positive.append(example)
    else:
        negative.append(example)

X1_pos = []
X2_pos = []
X1_neg = []
X2_neg = []
Y_pos = []
Y_neg = []
for i in positive:
    X1_pos.append(i[0])
    X2_pos.append(i[1])
    Y_pos.append(1)

for i in negative:
    X1_neg.append(i[0])
    X2_neg.append(i[1])
    Y_neg.append(0)

plt.plot(X1_pos, X2_pos, 'b.')
plt.plot(X1_neg, X2_neg, 'r.')
plt.axis([0, 1, 0, 1])
plt.show()

x1 = cl_train1[:,[0]]
x2 = cl_train1[:,[1]]
y = cl_train1[:,[2]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1_pos, X2_pos, Y_pos, c='b', marker='o')
ax.scatter(X1_neg, X2_neg, Y_neg, c='r', marker='o')


ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.show()
'''
