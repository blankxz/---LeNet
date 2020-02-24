import os
import torch
import cv2 as cv
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GaussConv2d(nn.Module):
    def __init__(self):
        super(GaussConv2d, self).__init__()
        gauss = [[0.03797616, 0.044863533, 0.03797616],
         [0.044863533, 0.053, 0.044863533],
         [0.03797616, 0.044863533, 0.03797616]]
        gauss_weight = torch.FloatTensor(gauss).unsqueeze(0).unsqueeze(0)
        self.gauss_weight = nn.Parameter(data=gauss_weight, requires_grad=False)
    
    def forward(self, x):
        gauss_res = F.conv2d(x.unsqueeze(1), self.gauss_weight, padding=0)
        return gauss_res
    
class ProfileConv2d(nn.Module):
    def __init__(self):
        super(ProfileConv2d, self).__init__()
        c = 0.5
        d = -0.3
        e = -5.5
        profile = [[c,c,c,c,c],
                  [c,d,d,d,c],
                  [c,d,e,d,c],
                  [c,d,d,d,c],
                  [c,c,c,c,c]]
        profile_weight = torch.FloatTensor(profile).unsqueeze(0).unsqueeze(0)
        self.profile_weight = nn.Parameter(data=profile_weight, requires_grad=False)
        
    def forward(self, x):
        profile_res = F.conv2d(x, self.profile_weight, padding=0)
        return profile_res

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        
        
        self.conv1 = nn.Sequential(
            GaussConv2d(),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            ProfileConv2d(),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(61,61),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(61,61),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(61,61)

 
    def forward(self, x):
        
        x1 = x[:, 0]
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.fc1(x3)
        x5 = self.fc2(x4)
        x6 = self.fc3(x5)
        return x3



def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将灰度化以及类型转化为Tensor
    '''
    im = Image.open(picture_dir)
    im = im.resize((256, 256))
    Lim = im.convert('L' )
    # Lim.show()
    return transform(Lim)

def non_max_sup(res):
    pass

def binarization(img,threshold):
    r = []
    for i in img:
        a = []
        for j in i:
            if j < threshold:
                a.append(-1)
            else:
                a.append(1)
        r.append(a)
    return r

def get_feature(pic_dir):
    # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
    transform = transforms.ToTensor()
    # 输入数据
    img = get_picture(pic_dir, transform)
    # 插入维度
    img = img.unsqueeze(0)
    img = img.to(device)

    # 特征输出
    net = FeatureExtractor().to(device)
    res = net(img).data.numpy()[0,0,:,:]

    i = 0
    # while i < 0.1:
    #     i += 0.001
    res = binarization(res,0.065)
    plt.imshow(res)
    # plt.savefig(str(i)+'.jpg')
    plt.show()
        
if __name__ == "__main__":
    
    get_feature('test_img/'+'img (13).jpg')