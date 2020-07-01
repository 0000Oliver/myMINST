# -*- coding: utf-8 -*-
import torch
import os
import numpy as np

import torch.nn.functional as F
import shutil
import torchvision as tv
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse
from resnet50 import resnet50
import matplotlib.pyplot as plt
import time
import yaml
import easydict as Edict


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()#3, 224, 224
        self.conv1 = nn.Conv2d(3,32,3)#32, 222, 222
        self.max_pool1 = nn.MaxPool2d(2)#32, 111, 111
        self.conv2 =nn.Conv2d(32,64,3)#64, 109, 109
        self.max_pool2 =nn.MaxPool2d(2)#64, 54, 54
        self.fc1 = nn.Linear(64*54*54,512)
        self.fc2 = nn.Linear(512,2)
    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = x.view(in_size,-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = torch.sigmoid(x)
        return x
# 创建模型
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()#3, 224, 224
        self.conv1 = nn.Conv2d(3, 32, 5)#32, 220, 220
        self.maxpool = nn.MaxPool2d(2, 2)#6, 110, 110
        self.conv2 = nn.Conv2d(32, 64, 5)#16, 106, 106
        self.fc1 = nn.Linear(64 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        in_size = x.size(0)
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(in_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = torch.sigmoid(x)
        return x

def build_optimizer(params, configs):
    if configs.Train.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params=params, lr=configs.Train.lr, momentum=configs.Train.momentum,
                            weight_decay=configs.Train.weight_decay)

    return optimizer

def adjust_lr(optimizer, configs):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= configs.Train.lr_decay

def train(configs):
    # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(configs.mean, configs.std)
    ])


    dataset_train = datasets.ImageFolder(configs.train_root,
                                         transform=data_transform)  # label是按照文件夹名顺序排序后存成字典
    dataset_test = datasets.ImageFolder(configs.test_root, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=configs.Train.batch_size,
        shuffle=configs.shuffle ,
        num_workers = configs.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=configs.Test.batch_size,
        shuffle=configs.shuffle,
        num_workers=configs.num_workers
    )
    model = resnet50(num_classes = configs.Num_Classes,pretrained=configs.Pretrained)
    optimizer = build_optimizer(model.parameters(),configs)
    cirterion = nn.CrossEntropyLoss()
    model.train()
    model_name = configs.backbone
    print(model_name)
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    for epoch in range(configs.Train.epochs):
        running_loss =0.0
        train_correct = 0
        train_total = 0
        if epoch > 0 and epoch // configs.Train.lr_decay_epochgap == 0:

            adjust_lr(optimizer, configs)
        for batch_idx,(data,target)in enumerate(train_loader):
            #print(batch_idx)
            # print(data.shape)
            data, target = data.to(device), target.to(device)#.float().reshape(target.shape[0], 1)
            optimizer.zero_grad()

            output = model(data)
            _, pred = torch.max(output.data, 1)
            #pred = torch.tensor([[1.0] if num[0] >= 0.5 else [0.0] for num in output]).to(device)
            train_correct += (pred == target.data).sum()
            # print(target.shape)
            # print(output.shape)
            loss = cirterion(output,target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += target.size(0)
        train_loss_list.append(running_loss/train_total)
        print(train_correct)
        print(train_total)
        train_accuracy_list.append(100*train_correct/train_total)
        print('train %d epoch loss: %.3f acc: %.3f ' %(
            epoch+1,running_loss/train_total,100*train_correct/train_total))#
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)#.float().reshape(target.shape[0], 1)
                output = model(data)
                test_loss += cirterion(output, target).item()
                _, pred = torch.max(output.data, 1)
                #pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
                test_correct += (pred == target.data).sum()
                test_total += target.size(0)
        test_loss_list.append(test_loss / test_total)
        test_accuracy_list.append(100 * test_correct / test_total)
        print('test %d epoch loss: %.3f acc: %.3f ' % (
            epoch+1,test_loss / test_total, 100 * test_correct / test_total))  #

    x1 = range(0, configs.Train.epochs)
    y1 = train_loss_list
    y2 = test_loss_list
    y3 = train_accuracy_list
    y4 = test_accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.plot(x1, y2, 'o-')

    plt.legend(["train_loss", "test_loss"])
    plt.title('Loss vs. epoches')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y3, '.-')
    plt.plot(x1, y4, '.-')
    plt.legend(["train_accuracy", "test_accuracy"])
    plt.xlabel('Accuracy vs. epoches')
    plt.ylabel('Accuracy')
    plt.show()

    plt.savefig(model_name+"_accuracy_loss.jpg")

    torch.save(model.state_dict(), '%s/catdognet_%s.pth' % (opt.outf, model_name))














# start = time.time()
# model = ConvNet()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# train(model,device,train_loader,test_loader,optimizer,epoch)
# end = time.time()
# print("costtime:",str(end-start))
#
# start = time.time()
# model = ConvNet2()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# train(model,device,train_loader,test_loader,optimizer,epoch)
# end = time.time()
# print("costtime:",str(end-start))
if __name__ == "__main__":
    # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 模型保存路径
    parser.add_argument('--net', default='./model/catdog.pth', help="path to netG (to continue training)")  # 模型加载路径
    opt = parser.parse_args()
    with open('./config.yaml') as f:
        configs = yaml.safe_load(f)
        configs = Edict.EasyDict(configs)
        print(configs)
        train(configs)

   #  start = time.time()
   # model = resnet50(True)
   #  optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
   #  train(model,device,train_loader,test_loader,optimizer,epoch)
   #  end = time.time()
   #  print("costtime:",str(end-start))
