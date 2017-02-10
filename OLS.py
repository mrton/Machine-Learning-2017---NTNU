import numpy as np
import matplotlib.pyplot as plt

# Importing data
test_1d = np.genfromtxt('./datasets/regression/reg-1d-test.csv', delimiter=",")
train_1d = np.genfromtxt('./datasets/regression/reg-1d-train.csv', delimiter=",")
test_2d = np.genfromtxt('./datasets/regression/reg-2d-test.csv', delimiter=",")
train_2d = np.genfromtxt('./datasets/regression/reg-2d-train.csv', delimiter=",")

# Calculating weights
def calc_weights(X,y):
    X_t = X.transpose()
    w = np.matmul( np.linalg.pinv(np.matmul(X_t,X)) , np.matmul(X_t,y) )
    return w

# Calculating mean square error
def mean_square_error(w,X,y):
    return (1 / X.shape[0] ) * (np.power((np.sum(np.subtract(np.matmul(X,w),  y))), 2))


X_train_1d = train_1d[0:,:1]
y_train_1d = train_1d[0:,1:2]
N = y_train_1d.size
X_train_1d = np.hstack([ np.ones((N,1)) , X_train_1d])
# Calculating weights with training 1D data
w = calc_weights(X_train_1d, y_train_1d)
# Calculating mean square error for 1D data
mse = mean_square_error(w,X_train_1d,y_train_1d)
print("training1D: " + str(mse))

X_test_1d = test_1d[0:,:1]
y_test_1d = test_1d[0:,1:2]
N = y_test_1d.size
X_test_1d = np.hstack([ np.ones((N,1)) , X_test_1d])
mse = mean_square_error(w,X_test_1d,y_test_1d)
print("test1D: " +str(mse))


X_train_2d = train_2d[0:,:2]
y_train_2d = train_2d[0:,2:3]
N = y_train_2d.size
X_train_2d = np.hstack([ np.ones((N,1)) , X_train_2d])
# Calculating weights with training 2D data
w = calc_weights(X_train_2d, y_train_2d)
# Calculating mean square error for 1D data
mse = mean_square_error(w, X_train_2d, y_train_2d)
print("training2D: " + str(mse))

X_test_2d = test_2d[0:,:2]
y_test_2d = test_2d[0:,2:3]
N_train2d = y_test_2d.size
X_test_2d = np.hstack([ np.ones((N_train2d,1)) , X_test_2d])
mse = mean_square_error(w, X_test_2d, y_test_2d)
print("test2D: " + str(mse))
