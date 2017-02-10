import numpy as np
import matplotlib.pyplot as plt

# REGRESSION

test_1d = np.genfromtxt('./datasets/regression/reg-1d-test.csv', delimiter=",")
train_1d = np.genfromtxt('./datasets/regression/reg-1d-train.csv', delimiter=",")
#test_2d = np.genfromtxt('./datasets/regression/reg-2d-test.csv', delimiter=",")
#train_2d = np.genfromtxt('./datasets/regression/reg-2d-train.csv', delimiter=",")

'''
X_train_2d = train_2d[0:,:2]
y_train_2d = train_2d[0:,2:3]
N_2d = y_train_2d.size
X_train_2d = np.hstack([ np.ones((N_2d,1)) , X_train_2d])
w = calc_weights(X_train_2d, y_train_2d)

X_train_1d = train_2d[0:,:1]
y_train_1d = train_2d[0:,1:2]
N_1d = y_train_1d.size
X_train_1d = np.hstack([ np.ones((N_1d,1)) , X_train_1d])
w = calc_weights(X_train_1d, y_train_1d)
print(w)
'''

X,Y = np.hsplit(test_1d,2)
print(X)

# For display in plot
x = []
y = []
for i in test_1d:
    x.append(i[0])
    y.append(i[1])

def calc_weights(X,y):
    X_t = X.transpose()
    w = np.matmul( np.linalg.pinv(np.matmul(X_t,X)) , np.matmul(X_t,y) )
    return w


def mean_square_error(w,X,y):
    # Should be the number of examples
    N = X.size / 2
    e = (1 / N) * np.power( np.absolute(np.matmul(X,w) - y) , 2)
    return e

ones = np.ones((240,1))
new_X = np.hstack([ones, X])

w = calc_weights(X,Y)
bias = np.array([[1]])
w = np.concatenate((bias,w))

mse = (w,new_X,y)

print(w)



plt.plot(x, y, 'b.')
plt.axis([0, 1, 0, 1])
plt.show()






#w = np.insert(w, 0, 10, axis=0)
