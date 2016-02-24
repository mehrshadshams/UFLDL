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


def prepare_data(images, labels, binary=True):
    X = loadMNIST_images(images)
    y = loadMNIST_labels(labels)

    if binary:
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
