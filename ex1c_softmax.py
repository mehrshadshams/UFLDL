import scipy.optimize
from scipy.optimize import fmin_cg
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import *
import mpmath as mp
import time

num_classifiers = 10


def softmax(X, theta):
    theta_x = theta.dot(X)
    h = np.exp(theta_x)
    probabilities = h / np.sum(h, axis=0)
    return probabilities


def reshape_theta(theta, n):
    return theta.reshape(num_classifiers, n) # k * n


def cost_regression(theta, X, y):
    n, m = X.shape
    theta = reshape_theta(theta, n)
    
    h = softmax(X, theta)
    J = -np.multiply(y, np.log(h)).sum() / m
    
#    print J

    errors = y - h
    g = -np.dot(errors, X.transpose()) / m
    g = np.ndarray.flatten(g)
    return [J, g]


def softmax_regression(X, y):
    n, m = X.shape
    theta = np.random.random((num_classifiers * n, 1))
    
    # any of these would work
#    rand = np.random.RandomState(int(time.time()))
#    theta = 0.005 * np.asarray(rand.normal(size = (num_classifiers * n, 1)))
    
    res = scipy.optimize.minimize(cost_regression, theta, args=(X, y), method='L-BFGS-B',
                                  jac = True, options = {'maxiter': 100, 'disp': True})
                                  
    theta = reshape_theta(res.x, n)
    return theta
    

def get_ground_truth(labels):
    # With the help of https://github.com/siddharth-agrawal/Softmax-Regression
    labels = np.array(labels).flatten()
    data   = np.ones(len(labels))
    indptr = np.arange(len(labels)+1)
    
    """ Compute the groundtruth matrix and return """
    
    ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
    ground_truth = np.transpose(ground_truth.todense())
        
    return ground_truth;
    

train_images = 'data/common/train-images-idx3-ubyte'
train_labels = 'data/common/train-labels-idx1-ubyte'
test_images = 'data/common/t10k-images-idx3-ubyte'
test_labels = 'data/common/t10k-labels-idx1-ubyte'

print('Loading train data...')
train_X, train_y = prepare_data(train_images, train_labels, binary=False)
train_X = train_X.transpose()

print('Loading test data...')
test_X, test_y = prepare_data(test_images, test_labels, binary=False)
test_X = test_X.transpose()

train_y = get_ground_truth(train_y)

theta = softmax_regression(train_X, train_y)

y_hat = softmax(test_X, theta)
predictions = np.zeros((test_y.shape[0], 1))
predictions[:, 0] = np.argmax(y_hat, axis=0)

correct = test_y[:,0] == predictions[:,0]
accuracy = correct.mean()

print('Accuracy: %2.1f' % (100 * accuracy))

r = np.random.randint(0, len(test_y))

print('Actual Label: %d, Predicted: %d' % (test_y[r,0], predictions[r, 0]))
x = test_X[:,r]
plt.imshow(x.reshape(28, 28))
