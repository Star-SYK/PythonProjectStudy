# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
import pandas as pd
import numpy as np


def linear1():
    """
    正规方程的优化方法对波士顿预测房价进行预测
    :return:
    """
    # 1) 获取数据
    # boston = load_boston()
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    boston = raw_df.values[1::2, 2]

    # 2) 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, boston, random_state=22)

    # 3) 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) 预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5) 得出模型
    print("正规方程权重系数为:\n", estimator.coef_)
    print("偏置为:\n", estimator.intercept_)

    # 6) 模型评估
    return None


def linear2():
    """
    梯度下降优化方法对波士顿预测房价进行预测
    :return:
    """
    # 1) 获取数据
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    boston = raw_df.values[1::2, 2]

    # 2) 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, boston, random_state=22)

    # 3) 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) 预估器
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 5) 得出模型
    print("梯度下降权重系数为:\n", estimator.coef_)
    print("偏置为:\n", estimator.intercept_)

    # 6) 模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差:\n", error)

    return None


def linear3():
    """
    岭回归方法对波士顿预测房价进行预测
    :return:
    """
    # 1) 获取数据

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    column_name = \
        ["CRIM",
         "ZN",
         "INDUS",
         "CHAS",
         "NOX",
         "RM",
         "AGE",
         "DIS",
         "RAD",
         "TAX",
         "PTRATIO",
         "B",
         "LSTAT",
         "MEDV"]
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    boston = raw_df.values[1::2, 2]

    # 2) 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, boston, random_state=22)

    # 3) 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) 预估器
    estimator = Ridge(alpha=0.5, max_iter=10000)
    estimator.fit(x_train, y_train)

    # 5) 得出模型
    print("岭回归权重系数为:\n", estimator.coef_)
    print("偏置为:\n", estimator.intercept_)

    # 6) 模型评估
    y_predict = estimator.predict(x_test)
    print("预测预测房价：\n", y_predict)
    from sklearn.metrics import mean_squared_error
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差:\n", error)

    return None


if __name__ == "__main__":
    linear1()
    linear2()
    linear3()
