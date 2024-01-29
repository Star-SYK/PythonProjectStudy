import numpy as np
import torch
import numpy


def createMatrix():
    """
    创建矩阵
    :return:
    """

    print("创建一个5*3的初始化矩阵:")
    x = torch.empty(5,3)
    print(x,"\n")

    print("创建一个5*3的随机矩阵:")
    x = torch.rand(5,3)
    print(x,"\n")

    print("创建一个5*3的零矩阵:")
    x = torch.zeros(5, 3,dtype = torch.long)
    print(x, "\n")

    print("创建一个自定义矩阵:")
    x = torch.tensor([5.5,0.3])
    print(x, "\n")

    print("创建一个自定义矩阵:")
    x = torch.tensor([5.5, 0.3])
    print(x, "\n")

    print("基于现有矩阵，重新创新一个结构相同数据不同的矩阵(数据取值于正态分布)")
    x = x.new_ones(5, 3, dtype=torch.double)
    print(x)
    x = torch.randn_like(x, dtype=torch.float)
    print(x, "\n")
    print("x矩阵的大小",x.size())

    return None
def matrixCompute():
    """
    矩阵的计算
    :return:
    """
    print("创建两个单位矩阵x、y,并让其相加")
    x = torch.ones(5,3)
    y = torch.ones(5,3)
    print("X+Y:\n",x+y)
    print("截取x的:\n")
    print("X:\n",x[1,:])
    return None

def matrixTransform():
    """
    1.矩阵的维度转换
    :return:
    """

    print("改变矩阵的维度")
    x = torch.rand(4,4)
    y = x.view(16)
    z = x.view(-1,8)
    print(x.size(),y.size(), z.size(),"\n")

    print("tensor转array")
    a = torch.ones(5)
    b = a.numpy()
    print("b:", b,"\n")

    print("array转tensor")
    a = np.ones(5)
    b = torch.from_numpy(a)
    print("b:", b,"\n")

    return None

if __name__ == "__main__":
    # createMatrix()
    # matrixCompute()
    matrixTransform()