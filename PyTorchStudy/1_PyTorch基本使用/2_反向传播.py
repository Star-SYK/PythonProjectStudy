import torch



def backpropagation():
    x = torch.randn(3,4,requires_grad=True)
    b = torch.randn(3,4,requires_grad=True)
    t = x + b
    y = t.sum()
    print("x:\n",x)
    print("b:\n",b)
    print("y:\n",y)
    y.backward()
    print("y的反向传播:",y)
    print("b:",b)
    print("b_grad:",b.grad)

    return None

if __name__ == "__main__":
    backpropagation()