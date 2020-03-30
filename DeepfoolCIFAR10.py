# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:49:01 2020

@author: GuoTong
"""
#Deepfool
import torch
import torchvision
from torchvision import transforms
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def show_images_diff(original_img,original_label,adversarial_img,adversarial_label,difference):
    plt.figure()
        
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')       
    plt.imshow(difference)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#定义模型结构
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

    
#获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#使用预测模式 主要影响droupout和BN层的行为
model = ResNet18().to(device)
model_path = "cifar_resnet18.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

for param in model.parameters():
    param.requires_grad = False

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
num_classes = 10
epochs = 10
overshoot = 0.02
softmax = nn.Softmax(dim=1)

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        ])

testdata = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
correct = 0
l2_max = 0.5
l2_total = 0
attack_total = 0
total = 10000

for i in range(total):
    #print(i)
    if i % 100 == 99:
        print(i-99,"--",i+1)
    img = testdata[i][0].unsqueeze(0).to(device)
    orig_label=testdata[i][1]
    
    new_output = model(img)
    scores=new_output.detach().cpu().numpy()[0]
    label=np.argmax(scores)
    
    if label != orig_label:
        continue
    
    #print("label={}".format(classes[orig_label]))
    
    orig = img.detach().cpu().numpy()[0]
    orig = orig.transpose(1,2,0)
    adv = img.detach().cpu().numpy()[0]
    adv = adv.transpose(1, 2, 0)
    difference = adv - orig
    
    img.requires_grad = True
    output = model(img)
    input_shape = img.cpu().detach().numpy().shape
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    
    attack_success = False
    
    for epoch in range(epochs):
    
        difference = adv - orig
        l2=np.linalg.norm(difference)       
        l2_percent=l2 / np.linalg.norm(orig)
        
        new_output = model(img)
        scores=new_output.detach().cpu().numpy()[0]
        label=np.argmax(scores)
        
        if l2_percent > l2_max:
            #print("attack failed")
            #print("epoch={} label={} {} score={:.4f} confidence={:.4f} l2={:.4f} l2_percent={:.4f}"
            #  .format(epoch,label,classes[label],scores[label],torch.max(softmax(new_output)).item(),l2,l2_percent))
            correct += 1
            break      
        #如果无定向攻击成功
        if label != orig_label:
            l2_total += l2_percent
            attack_total += 1
            #print("epoch={} label={} {} score={:.4f} confidence={:.4f} l2={:.4f} l2_percent={:.4f}"
            #  .format(epoch,label,classes[label],scores[label],torch.max(softmax(new_output)).item(),l2,l2_percent))
            break
       
        pert = np.inf
        output[0, orig_label].backward(retain_graph=True)
        grad_orig = img.grad.detach().cpu().numpy().copy()
    
        for k in range(1, num_classes):
            
            if k == orig_label:
                continue
            
            #梯度清零
            zero_gradients(img)
            
            output[0, k].backward(retain_graph=True)
            cur_grad = img.grad.detach().cpu().numpy().copy()
    
            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (output[0, k] - output[0, orig_label]).data.cpu().numpy()
    
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
    
            # 选择pert最小值
            if pert_k < pert:
                pert = pert_k
                w = w_k
                
        # 计算 r_i 和 r_tot
        r_i =  (pert+1e-8) * w / np.linalg.norm(w)     
        r_tot = np.float32(r_tot + r_i)
        
        #注意这里如果使用detach()则会报错，无法在改变img的值后求导
        img.data = img.data + (1+overshoot) * torch.from_numpy(r_tot).to(device)
        
        adv=img.detach().cpu().numpy()[0]
        adv = adv.transpose(1, 2, 0)
    
    # #(-1,1)  -> (0,1)
    difference = difference * std + mean
    difference=difference / abs(difference).max()/2.0+0.5  
    difference = np.clip(difference, 0, 1)
    
    orig = orig * std + mean
    orig = np.clip(orig, 0, 1)
    
    adv = (adv * std) + mean
    adv = np.clip(adv, 0, 1)
    
    #show_images_diff(orig,orig_label,adv,label,difference)

print("Deepfool test accuracy rate: {:.4f}".format(correct/(total)))
print("average l2_percentage : {}".format(l2_total/attack_total))
with open("DeepfoolCIFAR10_log.txt","w") as f:
    f.write("Deepfool test accuracy rate: {:.4f}".format(correct/(total)))
    f.write("average l2_percentage : {}".format(l2_total/attack_total))
    f.write("\n")