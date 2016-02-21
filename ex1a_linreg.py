import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler


def flatten(x, theta):
    return np.hstack((x.flatten('F'), theta.flatten('F')))


def unfold(params, m, n):
    x = params[0:(m * n)].reshape(m, n)
    theta = params[(m * n):].reshape(1, n)

    return (x, theta)


def gradientDescent(X, y, theta, alpha, iterations):
    (m, n) = X.shape
    for i in range(0, iterations):
        theta = np.multiply(theta, -alpha / m * ((X.dot(theta) - y).T.dot(X)).T)

    return theta


def gradient(theta, x, Y, m, n):
    h = x.dot(theta.T)

    errors = (h - Y).reshape(m, 1)

    g = x.T.dot(errors)
    return np.ndarray.flatten(g)


def cost_function(x, theta, Y):
    err = x.dot(theta.T) - Y
    J = 0.5 * (np.power(err, 2)).sum()

    return J


def plot(y, y_hat):
    y2 = y.copy()
    y2.sort()

    y_hat2 = y_hat.copy()
    y_hat2.sort()

    m = y2.shape[0]

    x = np.arange(m)
    area = np.pi * 10**2
    plt.scatter(x, y2, marker='x', color='blue', label='actual', s=area)
    plt.scatter(x, y_hat2, marker='x', color='red', label='predicted', s=area)

    plt.xlim([0, m])
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def linear_regression(theta, x, y, m, n):
    return cost_function(x, theta, y)


def main():
    data = np.genfromtxt('housing.csv', delimiter=',')

    data = np.hstack((np.ones((data.shape[0], 1)), data))

    # indexes = np.random.permutation(data.shape[0])

    # data = data[indexes, :].astype(float)

    c = 400

    train_x = data[:-1, :c].T
    train_y = data[-1, :c]

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(train_x)
    y_std = sc_y.fit_transform(train_y)

    m, n = train_x.shape
    train_y = train_y.reshape(m, 1)

    test_x = data[:-1, c + 1:].T
    test_y = data[-1, c + 1:]
    test_y = test_y.reshape(test_y.shape[0], 1)

    theta = np.random.random(n).reshape(n, 1)

    res = fmin_cg(linear_regression, theta, fprime=gradient, args=(X_std, y_std, m, n), maxiter=200, disp=True)

    Theta = res
    print('Theta: %s' % str(Theta))

    actual_prices = y_std
    predicted_prices = X_std.dot(Theta.T).reshape(train_x.shape[0], 1)

    train_rms = math.sqrt(np.power(predicted_prices - actual_prices, 2).mean())
    print('RMS training error: %f' % (train_rms))

    test_x_std = sc_x.transform(test_x)
    test_y_std = sc_y.transform(test_y)
    actual_prices = test_y_std
    predicted_prices = test_x_std.dot(Theta.T).reshape(test_x_std.shape[0], 1)

    test_rms = math.sqrt(np.power(predicted_prices - actual_prices, 2).mean())
    print('RMS testing error: %f' % (test_rms))

    plot(actual_prices, predicted_prices)


if __name__ == '__main__':
    main()
