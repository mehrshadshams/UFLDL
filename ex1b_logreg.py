from scipy.optimize import fmin_cg
import numpy as np
from load_mnist import *


def sigmoid(z):
    return 1.0 / (1 + np.power(np.e, -z))


def gradient(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta.T)).reshape(m, 1)
    errors = 1.0/m * (h - y).reshape(m, 1)
    g = X.T.dot(errors)
    return np.ndarray.flatten(g)


def cost_regression(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta.T)).reshape(m, 1)
    J = 1.0/m * -(y.T.dot(np.log(h)) + (1 - y.T).dot(np.log(1 - h))).sum()
    # print(J)
    return J


def logistic_regression(X, y):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    m, n = X.shape

    theta = np.random.random(n).reshape(1, n)

    res = fmin_cg(cost_regression, theta, fprime=gradient,
                  args=(X, y), maxiter=200, disp=True)

    theta = res
    # print('Theta: %s' % str(theta))
    return theta


def get_accuracy(theta, test_X, test_y):
    m, n = test_X.shape
    test_X = np.hstack((np.ones((m, 1)), test_X))
    h = sigmoid(test_X.dot(theta.T)).reshape(m, 1)
    correct = ((test_y == 1) == (h > 0.5)).sum(dtype=np.float32)
    return correct / m


train_images = 'data/common/train-images-idx3-ubyte'
train_labels = 'data/common/train-labels-idx1-ubyte'
test_images = 'data/common/t10k-images-idx3-ubyte'
test_labels = 'data/common/t10k-labels-idx1-ubyte'

print('Loading train data...')
train_X, train_y = prepare_data(train_images, train_labels)

print('Loading test data...')
test_X, test_y = prepare_data(test_images, test_labels)

theta = logistic_regression(train_X, train_y)
accuracy = get_accuracy(theta, test_X, test_y)

print('Accuracy: %2.1f' % (100 * accuracy))
