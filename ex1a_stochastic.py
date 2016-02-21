import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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


def shuffle(X, y):
    m = X.shape[0]
    indexes = np.random.permutation(m)
    X = X[indexes, :]
    y = y[indexes]

    return (X, y)


def normalize(sc_x, sc_y, X, y):
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)

    return (X_std, y_std)


def add_bias(X):
    m = X.shape[0]
    return np.hstack((np.ones((m, 1)), X))


def main():
    data = np.genfromtxt('housing.csv', delimiter=',')

    # data = np.hstack((np.ones((data.shape[0], 1)), data))

    c = 400

    train_x = data[:-1, :c].T
    train_y = data[-1, :c]

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std, y_std = normalize(sc_x, sc_y, train_x, train_y)

    X_std = add_bias(X_std)
    m, n = X_std.shape

    train_y = train_y.reshape(m, 1)

    test_x = data[:-1, c + 1:].T
    test_y = data[-1, c + 1:]
    test_y = test_y.reshape(test_y.shape[0], 1)

    test_x_std, test_y_std = normalize(sc_x, sc_y, test_x, test_y)
    test_x_std = add_bias(test_x_std)

    theta = np.random.random(n).reshape(n, 1)

    cost_ = []
    alpha = 0.01
    for iter in range(0, 35):
        X, y = shuffle(X_std, y_std)

        costs = []
        for xi, yi in zip(X, y):
            h = np.dot(xi, theta)

            err = yi - h

            theta += alpha * (xi * err).reshape(n, 1)
            # theta[0] += alpha * err
            cost = 0.5 * err ** 2
            costs.append(cost)

        avg_cost = sum(costs) / len(y)
        cost_.append(avg_cost)

    actual_prices = test_y_std
    predicted_prices = test_x_std.dot(theta)

    plot(actual_prices, predicted_prices)

    plt.plot(range(1, len(cost_) + 1), cost_, marker='o')
    plt.show()


if __name__ == '__main__':
    main()
