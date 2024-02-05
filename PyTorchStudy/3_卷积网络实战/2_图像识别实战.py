import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
import warnings
import time
import random
import sys
import copy
import json
from PIL import Image


def laod_data():
    """
    加载数据
    :return:
            dataloaders:批次数据
            cat_to_name:标签映射
    """
    # 数据增强
    train_compose = transforms.Compose([
        transforms.RandomRotation(45),  # 随机旋转，（-45,45）范围内选择旋转角度
        transforms.CenterCrop(224),  # 从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，选择一个概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道RGB
        transforms.ToTensor(),  # 数据转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
    ])

    valid_compose = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms = {"train": train_compose, "valid": valid_compose}

    # 加载数据
    batch_size = 6
    data_dir = "./flower_data/"
    data_keys = ["train", "valid"]
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_keys}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   data_keys}
    dataset_sizes = {x: len(image_datasets[x]) for x in data_keys}
    class_names = image_datasets["train"].classes

    print("训练和验证数据大小:", dataset_sizes)
    print("标签名称:", class_names)

    # 读取标签对应的实际名称
    with open(data_dir + "cat_to_name.json", "r") as label_json_file:
        cat_to_name = json.load(label_json_file)
    print(cat_to_name)

    return dataloaders, cat_to_name, class_names


def im_convert(tensor):
    """
    将tensor数据转换成图片
    :param tensor:
    :return:
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def show_image_data(dataloaders, cat_to_name, class_names):
    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2

    dataiter = iter(dataloaders["valid"])
    inputs, classes = dataiter.__next__()
    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.set_title(cat_to_name[str(class_names[classes[idx]])])
        plt.imshow(im_convert(inputs[idx]))

    plt.show()


def set_parameter_requires_grad(model, feture_extracting):
    """
    冻结模型的梯度更新
    :param model:  模型
    :param feture_extracting: 是否冻结
    :return:
    """
    if feture_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(weights=None)
        model_ft = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_model(model, device, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=""):
    since = time.time()
    best_acc = 0

    val_acc_history = []      # 验证集的准确率
    train_acc_history = []    # 训练集的准确率
    train_losses = []         # 训练集的损失
    valid_losses = []         # 验证集的损失
    LRs = [optimizer.param_groups[0]["lr"]]

    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict()) # 拷贝最佳模型权重
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 训练和验证
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == "train"):
                    if is_inception and phase == "train":
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:# resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.indices == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print("Time elapsed {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, filename)

            if phase == "valid":
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                # scheduler.step(epoch_loss)
                scheduler.step()
            elif phase == "train":
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


if __name__ == "__main__":
    dataloaders, cat_to_name, class_names = laod_data()
    # show_image_data(dataloaders, cat_to_name, class_names)
    model_name = "resnet"
    feature_extract = True
    model_fileName = "checkpoint.pth"
    model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print("CUDA is not available.  Traning on CPU...")
    else:
        print("CUDA is available!  Training on GPU ...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # 是否训练所有层
    params_to_update = model_ft.parameters()
    print("Param to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # 优化设置
    optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    criterion = nn.NLLLoss()

    (model_ft,
     val_acc_history,
     train_acc_history,
     valid_losses,
     train_losses,
     LRs) = train_model(model_ft,device,dataloaders,criterion, optimizer_ft,num_epochs=20,is_inception=(model_name == "inception"),filename=model_fileName)


    for param in model_ft.parameters():
        param.requires_grad = True

    # 再继续训练所有的参数，学习率调小一点
    optimizer= optim.Adam(params_to_update,lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

    # 损失函数
    criterion = nn.NLLLoss()
    # 加载权重
    checkpoint = torch.load(model_fileName)
    best_acc = checkpoint["best_acc"]
    model_ft.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    (model_ft,
     val_acc_history,
     train_acc_history,
     valid_losses,
     train_losses,
     LRs) = train_model(model_ft, device, dataloaders, criterion, optimizer_ft, num_epochs=20,
                        is_inception=(model_name == "inception"), filename=model_fileName)


