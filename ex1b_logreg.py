from scipy.optimize import fmin_cg
import numpy as np
import struct


def loadMNIST_images(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2051, "Bad magic number in %s" % (filename)

        num_images = struct.unpack('>I', (f.read(4)))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', (f.read(4)))[0]

        buffer_ = f.read()

        images = np.ndarray(
            shape=(num_rows * num_cols * num_images), dtype='B', buffer=buffer_)
        images = images.reshape((num_images, num_rows * num_cols))
        return images.astype(float) / 255.0


def loadMNIST_labels(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2049, "Bad magic number in %s" % (filename)

        num_labels = struct.unpack('>I', f.read(4))[0]

        buffer_ = f.read()

        labels = np.ndarray(shape=(num_labels, 1), dtype='B', buffer=buffer_)
        return labels


def prepare_data(images, labels):
    X = loadMNIST_images(images)
    y = loadMNIST_labels(labels)

    indices = np.where(y == [0, 1])[0]
    y = y[indices]
    X = X[indices]

    indices = np.random.permutation(np.arange(y.shape[0]))
    y = y[indices]
    X = X[indices]

    # r = np.random.randint(0, y.shape[0])
    # print('Label: %d' % (y[r]))
    # plt.imshow(X[r].reshape(28, 28))

    m, n = X.shape
    m_ = np.mean(X, axis=1).reshape(m, 1)
    std = np.std(X, axis=1).reshape(m, 1)

    X = np.subtract(X, m_).astype(np.float64)
    X = np.divide(X, std + 0.1)

    return (X, y)


def sigmoid(z):
    return 1.0 / (1 + np.power(np.e, -z))


def gradient(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta.T)).reshape(m, 1)
    errors = (h - y).reshape(m, 1)
    g = X.T.dot(errors)
    return np.ndarray.flatten(g)


def cost_regression(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta.T)).reshape(m, 1)
    errors = (h - y).reshape(m, 1)
    J = 0.5 * (np.power(errors, 2)).sum()
    return J


def logistic_regression(X, y):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    m, n = X.shape

    theta = np.random.random(n).reshape(1, n)

    res = fmin_cg(cost_regression, theta, fprime=gradient,
                  args=(X, y), maxiter=200, disp=True)

    theta = res
    # print 'Theta: %s' % str(theta)
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
