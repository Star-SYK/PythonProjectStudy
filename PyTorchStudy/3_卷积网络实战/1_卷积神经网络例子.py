import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np




# 构建模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

# 准确率作为评估标准
def accuracy(predictions,labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

if __name__ == "__main__":

    intput_size = 28  # 图像的总尺寸
    num_classes = 10  # 标签的种类数
    num_epochs = 3  # 训练的总循环周期
    batch_size = 64  # 一个批次的大小 ，64张图

    # 训练集
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)

    # 测试集
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

    # 构建batch数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # 获取模型
    net = CNN()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001) # 定义优化器，普通的随机梯度下降算法

    # 开始训练循环
    for epoch in range(num_epochs):
        # 当前epoch的结果保存下来
        train_rights  = []

        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(output,target)
            train_rights.append(right)

            if batch_idx % 100 == 0:
                net.eval()
                val_rights = []

                for (data, target)  in test_loader:
                    output = net(data)
                    right = accuracy(output,target)
                    val_rights.append(right)


                # 准确率计算
                train_rate = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]) )
                val_rate = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]) )

                print("train_rights:", train_rights)
                print("val_right:", val_rights)
                print("train_rate:", train_rate)
                print("val_rate:", val_rate)

                print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                    epoch,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data,
                    100. * train_rate[0].numpy() / train_rate[1],
                    100. * val_rate[0].numpy() / val_rate[1],
                ))

