# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:53:57 2020

@author: GuoTong
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_images_diff(original_img,original_label,adversarial_img,adversarial_label,difference):
    plt.figure()
        
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img,cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')   
     
    plt.imshow(difference,cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 网络架构
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
    
#使用预测模式 主要影响droupout和BN层的行为
model = Alexnet().to(device)
model_path = "./mnist_alexnet.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

for param in model.parameters():
    param.requires_grad = False

mean = (0.1307,)
std = (0.3081,)

num_classes = 10
#c的初始值
initial_const=10
#adam的最大迭代次数 论文中建议10000次
max_iterations=500
#adam学习速率
learning_rate=0.05
#二分查找最大次数
binary_search_steps=5
#k值
k=40

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        ])

testdata = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform_test)

total = 5
correct = 0
l2_total = []
for i in range(total):
    img = testdata[i][0].unsqueeze(0).to(device)
    orig_label = testdata[i][1]
    
    outputs = model(img)
    predicted = torch.max(outputs.data, 1)[1]
    # 如果是本身就分类错误的样本，就不攻击 这里只攻击分类正确的样本
    if predicted.cpu() != orig_label:
        continue
    
    boxmin = torch.min(img).item()
    boxmax = torch.max(img).item()
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    
    start_time = time.time()
    for target in range(10):
        
        if target == orig_label:
            continue
        
        lower_bound = 0
        upper_bound = 1e10
        
        c = initial_const
        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = np.zeros(img.shape)
        
        for outer_step in range(binary_search_steps):
            
            print("o_bestl2={} c={}".format(o_bestl2,c)  )
            tlab=torch.from_numpy(np.eye(num_classes)[target]).to(device).float()
            #把原始图像转换成图像数据和扰动的形态
            #反双曲正切 定义域(-1,1) 值域(-∞，+∞) 乘以0.99999保证值不为1
            timg = torch.from_numpy(np.arctanh((img.cpu().numpy() - boxplus) / boxmul * 0.999999)).to(device).float()
            modifier=torch.zeros_like(timg).to(device).float()
            #图像数据的扰动量梯度可以获取
            modifier.requires_grad = True
            
            optimizer = torch.optim.Adam([modifier],lr=learning_rate)
            
            for iteration in range(1,max_iterations+1):
                optimizer.zero_grad()
                
                #定义新的输入
                #变换到符合模型输入的范围
                newimg = torch.tanh(modifier + timg) * boxmul + boxplus
                output = model(newimg)
                
                #定义cw的损失函数
                # loss2 为原始图片和对抗样本的l2范数
                loss2 = torch.dist(newimg, (torch.tanh(timg) * boxmul + boxplus), p=2)
                
                #real 为 目标类的logits值 other为其他类最大的logits值  
                real = (output*tlab)[0,target]
                other = torch.max( (1 - tlab)*output - (tlab*10000) )
                #目标攻击
                loss1 = other - real + k
                loss1 = torch.clamp(loss1, min=0)
                loss1 = c * loss1
                
                loss = loss1 + loss2
                loss.backward(retain_graph = True)
                
                optimizer.step()
                
                l2 = loss2
                sc = output.detach().cpu().numpy()
                
                if l2 < o_bestl2 and np.argmax(sc) == target:
                    o_bestl2 = l2.item()
                    o_bestscore = np.argmax(sc)
                    o_bestattack = newimg.detach().cpu().numpy()
            
            old_c = -1
            if o_bestscore == target:
                #攻击成功 减小c
                print("attack success l2={} target_label={}".format(o_bestl2,target))
                upper_bound = min(upper_bound,c)
                if upper_bound < 1e9:
                    old_c = c
                    c = (lower_bound + upper_bound) / 2
            else:
                lower_bound = max(lower_bound,c)
                old_c = c
                if upper_bound < 1e9:
                    c = (lower_bound + upper_bound) / 2
                else:
                    c *= 10
            print("outer_step={} c {}->{}".format(outer_step,old_c,c))
        if o_bestscore == orig_label:
            correct += 1
        
        if o_bestscore == target:
            l2_total.append(o_bestl2)
        orig_img = img.detach().cpu().numpy().reshape(28, 28)
        adv_img = o_bestattack.reshape(28, 28)
        difference = adv_img - orig_img
        
        orig_img = orig_img * std + mean
        adv_img = adv_img * std + mean
        difference = difference * std + mean
        
        orig_img = np.clip(orig_img, 0, 1)  
        adv_img = np.clip(adv_img, 0 ,1)  
        difference = difference / abs(difference).max() / 2.0 +0.5          
        show_images_diff(orig_img, orig_label, adv_img, target, difference)
        
        
    print(" {}  {} ms".format(i,time.time() - start_time))     

print("CW test accuracy rate: {:.4f}".format(correct/(total*9)))
print("average l2:{} l2_percentage:{}".format(np.mean(l2_total),np.mean(l2_total)/np.linalg.norm(img.detach().cpu().numpy()[0].transpose(1,2,0))))