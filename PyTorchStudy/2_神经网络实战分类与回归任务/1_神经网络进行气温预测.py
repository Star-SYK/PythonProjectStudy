import datetime

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing

# 忽略警告
warnings.filterwarnings("ignore")

# 加载数据
features = pd.read_csv("./data/temps.csv")
print("数据维度:",features.shape)

# 数据处理(特征工程)
years = features["year"]
months = features["month"]
days = features["day"]

# 日期格式处理
dates = [str(int(year)) + "-" + str(int(month)) + "-" + str(int(day)) for year,month,day in zip(years, months, days) ]
dates = [datetime.datetime.strptime(date,"%Y-%m-%d") for date in dates]

# print("处理后的日期数据：",dates)
draw_chart(dates,features)

# 独热编码
features = pd.get_dummies(features)
# print(features)

# 标签
labels = np.array(features["actual"])

# 特征中去除标签
features = features.drop("actual",axis = 1)
features_list = list(features.columns)

# 转换合适的格式
features = np.array(features)

# 无量纲化
input_features = preprocessing.StandardScaler().fit_transform(features)

def draw_chart(x_datas,y_datas):
    """
    绘制图表
    :param x_datas: X轴数据
    :param y_datas: Y轴数据
    :return:
    """
    # 设置图表布局
    plt.style.use("fivethirtyeight")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.autofmt_xdate(rotation=45)

    # 标签值
    ax1.plot(x_datas, y_datas["actual"])
    ax1.set_xlabel("")
    ax1.set_ylabel("Temperature")
    ax1.set_title("Max Temp")

    ax2.plot(x_datas, y_datas["temp_1"])
    ax2.set_xlabel("")
    ax2.set_ylabel("Temperature")
    ax2.set_title("Previous Max Temp")

    ax3.plot(x_datas, y_datas["temp_2"])
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Temperature")
    ax3.set_title("Two Days Prior Max Temp")

    ax4.plot(x_datas, y_datas["friend"])
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Temperature")
    ax4.set_title("Friend Estimate")

    plt.tight_layout(pad=2)
    # 显示图表
    plt.show()

def temperature_prediction_model():
    """
    温度预测模型(采用复杂模式，注重了解神经网络模式)
    :return:
    """
    # 特征数据转换为张量
    x = torch.tensor(input_features,dtype = float)
    y = torch.tensor(labels,dtype = float)

    # 权重参数的初始化
    weights = torch.randn((14,128),dtype = float, requires_grad = True)
    biases = torch.randn(128,dtype = float, requires_grad = True)
    weights2 = torch.randn((128,1),dtype = float, requires_grad = True)
    biases2 = torch.randn(1,dtype = float, requires_grad = True)

    learning_rate = 0.001
    losses = []
    for i in range(1000):
        # 计算隐层
        hidden= x.mm(weights) + biases
        # 假如激活函数
        hidden = torch.relu(hidden)
        # 预测结果
        predictions = hidden.mm(weights2) + biases2
        # 通过计算损失
        loss = torch.mean((predictions -y) ** 2)
        losses.append(loss.data.numpy())

        if i % 100 == 0:
            print("loss: ",loss)

        # 反向传播计算
        loss.backward()

        # 更新参数
        weights.data.add_(- learning_rate * weights.grad.data)
        biases.data.add_(- learning_rate * biases.grad.data)
        weights2.data.add_(- learning_rate * weights2.grad.data)
        biases2.data.add_(- learning_rate * biases2.grad.data)


        # 每次迭代都得清空
        weights.grad.data.zero_()
        biases.grad.data.zero_()
        weights2.grad.data.zero_()
        biases2.grad.data.zero_()

def simple_temperature_prediction_model():
    """
    简易式温度预测模型
    :return:
    """
    input_size = input_features.shape[1]
    hidden_size = 128
    output_size = 1
    batch_size = 16
    my_nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size),
    )





if __name__ == "__main__":
    temperature_prediction_model()
