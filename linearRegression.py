import numpy as np
import matplotlib.pyplot as plt


def computerCost(X, y, theta):
    m = len(y)
    J = 0

    J = (np.transpose(X * theta - y) * (X * theta - y)) / (2 * m)  # 两个列向量不能相乘，要转置变化为线代运算！！！
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)

    temp = np.matrix(np.zeros((n, num_iters)))  # 用矩阵是为了进行线性代数的运算点积
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = np.dot(X, theta)
        temp[:, i] = theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))
        theta = temp[:, i]
        J_history[i] = computerCost(X, y, theta)
    return theta, J_history


def featureNormalize(X):
    X_norm = np.array(X)

    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X_norm, 0)  # 求每一列的平均值
    sigma = np.std(X_norm, 0)  # 求每一列的标准差
    for i in range(X.shape[1]):
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma


def plotJ(J_history, num_iters):
    x = np.arange(1, num_iters + 1)
    plt.plot(x, J_history)
    plt.xlabel(u"Number of iterations", size=14)
    plt.ylabel(u"Cost value", size=14)
    plt.show()


def loadtxtAndcsv_data(file_name, split, dataType):
    return np.loadtxt(file_name, delimiter=split, dtype=dataType)
# 以numpy打开csv文件，delimiter是间隔的意思，dtype是数据类型对象


def linearRegression(alpha, num_iters):
    # 加载数据
    data = loadtxtAndcsv_data("boston_house_prices.csv", ",", np.float64)
    X = data[:, 0:-1]  # 前面的冒号代表的是行的维度，即从第一行到最后一行
    y = data[:, -1]  # 每一行的最后一个元素
    m = len(y)
    col = data.shape[1]  # shape()函数 无括号，输出的是矩阵的行列，输入0，代表行数，1代表列数
    X, mu, sigma = featureNormalize(X)

    X = np.hstack((np.ones((m, 1)), X))  # m行1列的1，hstack是从水平堆叠（列方向），vstack是行方向
    theta = np.zeros((col, 1))
    y = y.reshape(-1, 1)  # 变换成一列的形式
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
    plotJ(J_history, num_iters)
    return mu, sigma, theta, J_history[-1]


def testlinearRegression():
    mu, sigma, theta, J_history = linearRegression(0.1, 200)
    print("\n theta={},\n Minimum cost={}".format(theta, J_history))


if __name__ == "__main__":
    testlinearRegression()
"""作为代码块时不运行"""
