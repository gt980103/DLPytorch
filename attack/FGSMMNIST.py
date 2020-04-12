# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:34:55 2020

@author: GuoTong
"""


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
            nn.AvgPool2d(3),
        )
        
        self.classifer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,256)
        x = self.classifer(x)
        return x


def fgsm(image, eps, image_grad):
    max_val = torch.max(image)
    min_val = torch.min(image)
    sign_grad = torch.sign(image_grad)
    adv_image = image + eps*sign_grad
    adv_image = torch.clamp(adv_image,min_val.item(),max_val.item())
    return adv_image
    


if __name__=="__main__":

    
    mean = (0.1307,)
    std = (0.3081,)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    # 导入数据，定义数据接口
    testdata  = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform_test)

    model = Alexnet().to(device)
    model_path = "./mnist_alexnet.pth"
    model.load_state_dict(torch.load(model_path)) # 加载模型
    # eval 之影响BN和Drop层的行为，还是会计算梯度，而torch.no_grad()则不会计算梯度，会使得雅可比矩阵计算失败
    model.eval()
    total = 10000
    correct = 0
    l2_percent_total = 0
    l2_total = 0
    attack_total = 0
    criterion = nn.CrossEntropyLoss()
    for i in range(total):
        if i % 500 == 499:
            print(i-499,"--",i+1)
        # 选择测试样本
        image = testdata[i][0].unsqueeze(0).to(device) # 测试样本特征
        label = torch.tensor([testdata[i][1]]) # 测试样本真实标签
        eps = 0.7
       
        image.requires_grad_()
        output = model(image)
        pred_label = torch.max(output,dim=1)
        
        if pred_label[1].cpu() != label:
            continue
        
        loss = criterion(output, label.to(device))
        model.zero_grad()
        loss.backward()
        
        image_grad = image.grad
        # 生成对抗性样本
        adv_image = fgsm(image, eps, image_grad)
        
        
        # 模型对对抗性样本的预测结果
        outputs = model(adv_image)
        predicted = torch.max(outputs, 1)[1]
        
        # 绘制对抗性样本的图像      
        orig_im = image.detach().cpu().numpy().reshape(28, 28)
        adv_image = adv_image.cpu().detach().numpy().reshape(28, 28)
        
        difference = adv_image - orig_im
        
        l2=np.linalg.norm(difference)       
        l2_percent=l2 / np.linalg.norm(orig_im)
        
        orig_im = orig_im*std + mean
        orig_im = np.clip(orig_im, 0, 1)     
        
        adv_image = adv_image*std + mean
        adv_image = np.clip(adv_image, 0 ,1)
        difference = adv_image - orig_im
    
        #(-1,1)  -> (0,1)
        difference=difference / abs(difference).max() / 2.0 +0.5
        #show_images_diff(orig_im, label, adv_image, predicted[0], difference)
        
        
        #print('测试样本扰动前的预测值：{}'.format(label.item()))
        #print('测试样本扰动后的预测值：{} l2:{} l2_percentage:{} '.format(predicted[0],l2,l2_percent))
        
        if predicted[0].cpu()==label:
            #print("攻击失败")
            correct += 1
        else:
            l2_percent_total += l2_percent
            l2_total += l2
            attack_total += 1
        
        
    print("FGSM test accuracy rate: {:.4f}".format(correct/(total)))
    print("average l2 : {}".format(l2_total/attack_total))
    print("average l2_percentage:{}".format(l2_percent_total/attack_total))
    with open("FGSMMNIST_log.txt","w") as f:
        f.write("FGSM test accuracy rate: {:.4f}".format(correct/(total)))
        f.write("average l2 : {}".format(l2_total/attack_total))
        f.write("average l2_percentage : {}".format(l2_percent_total/attack_total))
        f.write("\n")