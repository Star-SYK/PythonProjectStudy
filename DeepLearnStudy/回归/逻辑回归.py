import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def logistic_regression():
    """
    (预测癌症分类)
    :return: 返回模型
    """

    # 1.加载数据
    path  = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(path, names=column_name)
    # 缺失数据处理
    data = data.replace(to_replace="?",value = np.nan)
    data.dropna(inplace=True)


    # 2.划分数据集
    x = data.iloc[:,1:-1]
    y = data["Class"]
    x_train,x_test,y_train,y_test = train_test_split(x,y)

    # 3.数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.预估器流程
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)
    print("逻辑回归权重系数为:\n", estimator.coef_)
    print("偏置为:\n", estimator.intercept_)

    # 5、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # roc曲线和auc指标
    y_true = np.where(y_test > 3 , 1,0)
    auc_value = roc_auc_score(y_true,y_predict)
    print("auc指标为：\n", auc_value)

    return estimator


if __name__ == "__main__":
    logistic_regression()