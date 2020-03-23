# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:57:02 2020

@author: GuoTong
"""


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time

class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet,self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        
        self.classifer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*3*3,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,256*3*3)
        x = self.classifer(x)
        return x

def evl_test_acc(test_loader ,model, device = None):
    if device is None and isinstance(model, torch.nn.Module):
        # 如果没指定device就使用model的device
        device = list(model.parameters())[0].device 
    with torch.no_grad():
        model.eval()
        total = 0.0
        correct = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predict = torch.max(outputs,1)
            total += labels.size(0)
            correct += (labels == predict).sum().item()
        model.train()
    return correct / total


batch_size = 256
lr = 0.1
epochs = 60

mean = (0.1307,)
std = (0.3081,)

transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=1, contrast=0.1, hue=0.5),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])    

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

train_set = torchvision.datasets.MNIST("data",train=True,transform=transform_train,download=True)
test_set = torchvision.datasets.MNIST("data",train=False,transform=transform_test,download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print("train on ",device)

model = Alexnet().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4)

start_time = time.time()
batch_size_time = time.time()

test_accs = []
train_accs = []
best_test_acc = 0.0
best_epoch = 0
for epoch in range(epochs):

    running_loss = 0.0
    train_total = 0.0
    train_correct = 0.0
    start = time.time()

    if epoch == 10:
        lr = lr * 0.5
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
    if epoch == 20:
        lr = lr * 0.5
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
      #optimizer = optim.Adam(model.parameters(), lr=lr)
    if epoch == 30:
        lr = lr * 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for i,data in enumerate(train_loader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #清空梯度
        optimizer.zero_grad()
        #正向传播 反向传播 更新参数
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        #记录损失
        running_loss += loss.item()
        
        #训练集数量 训练集正确数量
        train_total += labels.size(0)
        _, predict = torch.max(output.data,1)
        train_correct += (labels == predict).sum().item() 
        
        #每50组batch 打印一次
        if i % 50 == 49:
            print('[%d, %5d] loss: %.5f train_acc: %.5f time: %.3f sec' % (epoch + 1, i + 1, running_loss / 50, train_correct / train_total ,time.time()-batch_size_time))
            running_loss = 0.0
            batch_size_time = time.time()
            
    #记录一次epoch的训练误差 测试误差 训练时间
    test_acc = evl_test_acc(test_loader, model)
    if test_acc > 0.98 and test_acc > best_test_acc:
        best_epoch = epoch
        best_test_acc = test_acc
        PATH = './mnist_alexnet.pth'
        torch.save(model.state_dict(), PATH)
    test_accs.append(test_acc)
    train_accs.append(train_correct / train_total)
    print('epoch %d, train acc: %.5f, test acc: %.5f, time: %.3f sec'
              % (epoch + 1,  train_correct / train_total, test_acc, time.time() - start))
    print("total time %.3f sec" % (time.time() - start_time))

print("best test accuracy is ",best_test_acc)

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
x = range(len(train_accs))

plt.ylim(0,1.0)
plt.yticks(np.linspace(0,1,11,endpoint=True))

plt.title("AlexNet-MNIST")
plt.ylabel("accuracy")
plt.xlabel("eopch")

plt.plot(x,train_accs,label="train_acc")
plt.plot(x,test_accs,label="test_acc")
plt.legend(loc = "lower right")

plt.annotate('%.4f' % best_test_acc ,
         xy=(best_epoch, best_test_acc), 
         xytext=(+30, +20), textcoords='offset points', fontsize=12,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.scatter([best_epoch,],[best_test_acc,], 10, color ='red')

plt.show()
plt.savefig('AlexNet-MNIST.png')