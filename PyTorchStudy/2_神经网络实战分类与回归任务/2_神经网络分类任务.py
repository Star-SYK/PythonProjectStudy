import pickle
import gzip
import torch
import torch.nn.functional as fun
from torch import nn
from torch import optim
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from matplotlib import pyplot



class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784,128)
        self.hidden2 = nn.Linear(128,256)
        self.out =nn.Linear(256,10)

    def forward(self,x):
        x = fun.relu(self.hidden1(x))
        x = fun.relu(self.hidden2(x))
        x = self.out(x)
        return x

def get_model():
    """
    模型获取
    :return:
    """
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(),lr=0.001)
def read_data(filePath):
    """
    读取数据
    :return: x_train,y_train,x_valid,y_valid
    """
    # 读取文件
    with gzip.open(filePath,"rb") as f:
        ((x_train, y_train),(x_valid, y_valid),_) = pickle.load(f,encoding="latin-1")
    # 返回转换成tensor后的数据
    return map(torch.tensor, (x_train, y_train, x_valid, y_valid))

def get_data(train_ds,valid_ds,bs):
    return (
        DataLoader(train_ds,batch_size = bs,shuffle = True),
        DataLoader(valid_ds,batch_size = bs*2)
    )

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(steps,model,loss_func,opt,train_dl,valid_dl):
    for step in range(steps):
        model.train();
        for xb,yb in train_dl:
            loss_batch(model,loss_func,xb,yb,opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print("当前step:"+str(step),"验证集损失："+str(val_loss))

if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid = read_data("data/mnist.pkl.gz")
    train_ds = TensorDataset(x_train,y_train)
    valid_ds = TensorDataset(x_valid,y_valid)
    train_dl, valid_dl = get_data(train_ds, valid_ds, 64)
    model, opt= get_model()
    loss_func = fun.cross_entropy
    fit(25, model, loss_func, opt, train_dl, valid_dl)

