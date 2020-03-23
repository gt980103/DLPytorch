import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)


import torchvision
import torchvision.transforms as transforms
#使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("train on ",device)

#设置CIFAR10的均值和方差
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
#设置批量大小
batch_size = 256
lr = 0.1
epochs = 120

#设置数据预处理 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])

#读取数据集 设置训练集和测试集
train_set = torchvision.datasets.CIFAR10(root='../data/data9154',train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='../data/data9154',train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

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

import torch.nn as nn
import torch.optim as optim
import time

# model = torchvision.models.resnet50(pretrained=True)
# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, 10)

model = ResNet18()
# model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
# model.avgpool = nn.AvgPool2d(4, stride=1)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#optimizer = optim.Adam(model.parameters(), lr=lr)

start_time = time.time()
each_epoch_time = time.time()

test_accs = []
train_accs = []
best_test_acc = 0.0
best_epoch = 0
for epoch in range(epochs):
    running_loss = 0.0
    train_total = 0.0
    train_correct = 0.0
    start = time.time()

    if epoch == 50:
        lr = lr * 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
    if epoch == 70:
        lr = lr * 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
      #optimizer = optim.Adam(model.parameters(), lr=lr)
    if epoch == 85:
        lr = lr * 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    if epoch == 100:
        lr = lr * 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        
    if epoch == 110:
        lr = lr * 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
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
            print('[%d, %5d] loss: %.5f train_acc: %.5f time: %.3f sec' % (epoch + 1, i + 1, running_loss / 50, train_correct / train_total ,time.time()-each_epoch_time))
            running_loss = 0.0
            each_epoch_time = time.time()
    #记录一次epoch的训练误差 测试误差 训练时间
    test_acc = evl_test_acc(test_loader, model)
    if test_acc > best_test_acc and test_acc > 0.93:
        best_test_acc = test_acc
        best_epoch = epoch
        PATH = './cifar_resnet18.pth'
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

plt.title("ResNet18-CIFAR10")
plt.ylabel("accuracy")
plt.xlabel("eopch")

plt.plot(x,train_accs,label="train_acc")
plt.plot(x,test_accs,label="test_acc")
plt.legend(loc = "upper left")

plt.annotate('%.4f' % best_test_acc ,
         xy=(best_epoch, best_test_acc), 
         xytext=(+30, +20), textcoords='offset points', fontsize=12,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.scatter([best_epoch,],[best_test_acc,], 10, color ='red')

plt.show()
plt.savefig('ResNet18-CIFAR10.png')
