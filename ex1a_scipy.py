import numpy as np

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


def main():
    data = np.genfromtxt('housing.csv', delimiter=',')

    data = np.hstack((np.ones((data.shape[0], 1)), data))

    # indexes = np.random.permutation(data.shape[0])

    # data = data[indexes, :].astype(float)

    c = 400

    train_x = data[:-1, :c].T
    train_y = data[-1, :c]

    m, n = train_x.shape
    train_y = train_y.reshape(m, 1)

    test_x = data[:-1, c + 1:].T
    test_y = data[-1, c + 1:]
    test_y = test_y.reshape(test_y.shape[0], 1)

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(train_x)
    y_std = sc_y.fit_transform(train_y)

    regr = linear_model.LinearRegression()
    regr.fit(X_std, y_std)

    print('Coefficients: \n', regr.coef_)

    test_x_std = sc_x.transform(test_x)
    test_y_std = sc_y.transform(test_y)
    test_pred = regr.predict(test_x_std)
    print test_pred.shape
    print test_y.shape
    print("Residual sum of squares: %.2f" % np.mean((test_pred - test_y_std) ** 2))
    print('Variance score: %.2f' % regr.score(test_x_std, test_y_std))

    x = np.arange(test_y.shape[0])
    area = np.pi * 10**2
    plt.scatter(x, test_y_std, marker='x', color='blue', label='actual', s=area)
    plt.scatter(x, test_pred, marker='x', color='red', label='predict', s=area)

    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
