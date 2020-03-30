# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:13:27 2020

@author: GuoTong
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

# 网络架构
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




# 计算雅可比矩阵，即前向导数
def compute_jacobian(model, image, num_features):

    output = model(image)
    classes = output.size()[1]
    #初始化雅可比矩阵
    jacobian = torch.zeros([classes, num_features]) # 对于MNIST数据集，是一个10*784的矩阵
    mask = torch.zeros(output.size())  # chooses the derivative to be calculated
    for i in range(classes):
        mask[:, i] = 1
        zero_gradients(image)
        # retain_graph=True 使得下次计算backward可以正常执行
        output.backward(mask.to(device), retain_graph=True)
        # copy the derivative to the target place
        jacobian[i] = image._grad.squeeze().view(-1, num_features).clone()
        mask[:, i] = 0  # reset

    return jacobian


# 计算显著图
def saliency_map(jacobian, target_index, increasing, search_space, nb_features):


    domain = torch.eq(search_space, 1).float()  # The search domain
    # the sum of all features' derivative with respect to each class
    all_sum = torch.sum(jacobian, dim=0, keepdim=True)
    target_grad = jacobian[target_index]  # The forward derivative of the target class
    others_grad = all_sum - target_grad  # The sum of forward derivative of other classes

    # this list blanks out those that are not in the search domain
    # Multiplying 2 here is to change the sign of the points that are not 
    # in the search field, so that they are all greater than 0 or less than 0
    # so they are removed in later operations
    
    #把搜索域中为0的点找出来，在increase_coef中为1
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float()
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float()
        
    
    increase_coef = increase_coef.view(-1, nb_features)  
    target_tmp = target_grad.clone()
    target_tmp -= increase_coef.cpu() * torch.max(torch.abs(target_grad)).cpu()
    # calculate sum of target forward derivative of any 2 features.
    # PyTorch will automatically extend the dimensions 利用广播机制
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)  
    # calculate sum of other forward derivative of any 2 features.
    others_tmp = others_grad.clone()
    others_tmp += increase_coef.cpu() * torch.max(torch.abs(others_grad)).cpu()
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    # zero out the situation where a feature sums with itself
    # 对角线上元素为自己与自己相加，为同一个点
    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte()

    # According to the definition of saliency map in the paper (formulas 8 and 9),
    # those elements in the saliency map that doesn't satisfy the requirement will be blanked out.
    if increasing:
        mask1 = torch.gt(alpha, 0.0)
        mask2 = torch.lt(beta, 0.0)
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)
    # apply the mask to the saliency map
    # 取mask1 mask2的交集，为取得的p q两点的搜索域，并把对角线元素置0
    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    # do the multiplication according to formula 10 in the paper
    if increasing:
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    else:
        saliency_map = torch.mul(torch.mul(torch.abs(alpha), beta), mask.float())
    # get the most significant two pixels
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    # get the position of p and q in the saliency map
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q


def jsma_attack(image, target, theta, gamma, model):

    max_val = torch.max(image)
    min_val = torch.min(image)
    
    var_sample = image.clone().detach().requires_grad_()
    

    var_target = torch.tensor([target])

    if theta > 0:
        increasing = True
    else:
        increasing = False

    num_features = int(np.prod(var_sample.shape[1:]))
    shape = var_sample.size()
    
    # perturb two pixels in one iteration, thus max_iters is divided by 2.0
    max_iters = int(np.ceil(num_features * gamma / 2.0))

    # masked search domain, if the pixel has already reached the top or bottom, we don't bother to modify it.
    search_domain = torch.ones_like(var_sample)
    
    search_domain = search_domain.view(num_features)
    
    output = model(var_sample)
    current = torch.max(output.data, 1)[1].item()

    iter = 0
    while (iter < max_iters) and (current != target) and (search_domain.sum() != 0):
        # calculate Jacobian matrix of forward derivative
        jacobian = compute_jacobian(model, var_sample, num_features)
        # get the saliency map and calculate the two pixels that have the greatest influence
        p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
        # apply modifications
        var_sample_flatten = var_sample.view(-1, num_features)
        var_sample_flatten[0, p1] += theta*(max_val - min_val)
        var_sample_flatten[0, p2] += theta*(max_val - min_val)

        new_sample = torch.clamp(var_sample_flatten, min=min_val.item(), max=max_val.item())
        new_sample = new_sample.view(shape)
        search_domain[p1] = 0
        search_domain[p2] = 0
        # var_sample = Variable(torch.tensor(new_sample), requires_grad=True)
        var_sample = new_sample.clone().detach().requires_grad_(True)
        output = model(var_sample)
        current = torch.max(output.data, 1)[1].item()
        iter += 1

    adv_samples = var_sample.data.cpu().numpy()
    return adv_samples


if __name__=="__main__":

    with open("log.txt","w") as f:
        f.write("log file")
        f.write("\n")
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    # 导入数据，定义数据接口
    testdata  = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

    model = ResNet18().to(device)
    model_path = "./cifar_resnet18.pth"
    model.load_state_dict(torch.load(model_path)) # 加载模型
    # eval 之影响BN和Drop层的行为，还是会计算梯度，而torch.no_grad()则不会计算梯度，会使得雅可比矩阵计算失败
    model.eval()
    total = 500
    correct = 0
    l0_total = 0
    attack_total = 0
    start = time.time()
    for i in range(total):
        print(i,time.time()-start)
        with open("log.txt","a") as f:
            f.write(str(i)+" "+str(time.time()-start)+"ms")
            f.write("\n")
        
        # 选择测试样本
        image = testdata[i][0].unsqueeze(0).to(device) # 测试样本特征
        label = torch.tensor([testdata[i][1]]) # 测试样本真实标签
        
        theta = 1.0 # 扰动值
        gamma = 0.032 # 最多扰动特征数占总特征数量的比例
        
        outputs = model(image)
        predicted = torch.max(outputs.data, 1)[1]
        
        # 如果是本身就分类错误的样本，就不攻击 这里只攻击分类正确的样本
        if predicted.cpu() != label:
            continue
        
        for target in range(10):
            
            #如果目标是真实标签，则跳过
            if target == label:
                continue
   
            # 生成对抗性样本
            adv_image = jsma_attack(image, target, theta, gamma, model)
        
            # 模型对对抗性样本的预测结果
            adv_image = torch.from_numpy(adv_image).requires_grad_().to(device)
            outputs = model(adv_image)
            predicted = torch.max(outputs.data, 1)[1]
            
            # 绘制对抗性样本的图像      
            orig_im = image.cpu().numpy()[0].transpose(1,2,0)
            adv_image = adv_image.cpu().detach().numpy()[0].transpose(1,2,0)
            difference = adv_image - orig_im
            l0=np.where(difference!=0)[0].shape[0]
            #l0 扰动
            l0_percent = l0 / (np.prod(orig_im.shape))
  
            orig_im = orig_im*std + mean
            orig_im = np.clip(orig_im, 0, 1)     
            
            adv_image = adv_image*std + mean
            adv_image = np.clip(adv_image, 0 ,1)
                  
            #(-1,1)  -> (0,1)
            difference=difference / abs(difference).max() / 2.0 +0.5
            #show_images_diff(orig_im, label, adv_image, predicted[0], difference) 
            
            #print('测试样本扰动前的预测值：{} {}'.format(label.item(),classes[label]))
            #print('测试样本扰动后的预测值：{} {} l0：{} l0_percentage:{}'.format(predicted[0],classes[predicted[0]],l0,l0_percent))
            # label 样本真实标签 target 目标标签 predicted[0]预测标签
            if predicted[0].cpu()==target:
                l0_total += l0
                attack_total += 1
            #else:
            #    print("目标攻击失败")
            
            if predicted[0].cpu()==label:
                correct += 1
                
    
    print("JSMA test accuracy rate: {:.4f}".format(correct/(total*9)))
    print("average l0:{} l0_percentage:{}".format(l0_total/attack_total,l0_total/attack_total/np.prod(orig_im.shape)))
    with open("log.txt","a") as f:
        f.write("JSMA test accuracy rate: {:.4f}".format(correct/(total*9)))
        f.write("\n")
        f.write("average l0:{} l0_percentage:{}".format(l0_total/attack_total,l0_total/attack_total/np.prod(orig_im.shape)))
