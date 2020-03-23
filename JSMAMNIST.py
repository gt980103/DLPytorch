# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:02:01 2020

@author: GuoTong
"""



import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    #把搜索域中为0的点找出来，在increase_coef中为2或-2
    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float()
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float()
        
    
    increase_coef = increase_coef.view(-1, nb_features)  
    target_tmp = target_grad.clone()
    target_tmp -= increase_coef.cpu() * torch.max(torch.abs(target_grad))
    # calculate sum of target forward derivative of any 2 features.
    # PyTorch will automatically extend the dimensions 利用广播机制
    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)  
    # calculate sum of other forward derivative of any 2 features.
    others_tmp = others_grad.clone()
    others_tmp += increase_coef.cpu() * torch.max(torch.abs(others_grad))
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
    outputs = model(var_sample)
    predicted = torch.max(outputs.data, 1)[1]
    print('测试样本扰动前的预测值：{}'.format(predicted[0]))

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
        print("epoch:{} p1:{} p2:{} label:{}".format(iter, p1, p2, current))
        iter += 1

    adv_samples = var_sample.data.cpu().numpy()
    return adv_samples


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
    
    index = 7777 # 选择测试样本
    image = testdata[index][0].unsqueeze(0).to(device) # 测试样本特征
    label = torch.tensor([testdata[index][1]]).to(device) # 测试样本真实标签
    
    theta = 1.0 # 扰动值
    gamma = 0.1 # 最多扰动特征数占总特征数量的比例
    target= 3 # 对抗性样本的标签

    # 生成对抗性样本
    adv_image = jsma_attack(image, target, theta, gamma, model)

    # 模型对对抗性样本的预测结果
    adv_image = torch.from_numpy(adv_image).requires_grad_().to(device)
    outputs = model(adv_image)
    predicted = torch.max(outputs.data, 1)[1]
    print('测试样本扰动后的预测值：{}'.format(predicted[0]))
    
    # 绘制对抗性样本的图像
    
    orig_im = image.cpu().numpy()[0].reshape(28, 28)
    orig_im = orig_im*std + mean
    orig_im = np.clip(orig_im, 0, 1)
    plt.figure()
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(orig_im,cmap='gray')
    plt.axis("off")
    
    adv_image = adv_image.cpu().detach().numpy()
    im = adv_image.reshape(28, 28)
    im = im*std + mean
    im = np.clip(im, 0 ,1)
    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = im - orig_im
    #(-1,1)  -> (0,1)
    difference=difference / abs(difference).max() / 2.0 +0.5
    plt.imshow(difference,cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    
    
    

'''
    plt.figure()
    for i in range(10):
        adv_image = jsma_attack(image, i, theta, gamma, model)
        im = adv_image.reshape(28, 28)
        plt.subplot(2,5,i+1)
        plt.axis('off')
        plt.title("Flag: " + str(i))
        plt.imshow(im, cmap='gray')
    plt.show()
    plt.savefig("test.png")
'''
