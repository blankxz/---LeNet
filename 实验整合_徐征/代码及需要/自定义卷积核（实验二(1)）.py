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

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
         # 第一层神经网络，包括卷积层、线性激活函数、池化层
        
        
        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0),  # input_size=(32*128*128)
            nn.ReLU(),            # input_size=(64*128*128)
            nn.MaxPool2d(2, 2)    # output_size=(64*64*64)
        )

        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(84, 62)
 
    def forward(self, x):
        c = 0.5
        d = -0.3
        e = -5.5
        kernel = [[c,c,c,c,c],
                  [c,d,d,d,c],
                  [c,d,e,d,c],
                  [c,d,d,d,c],
                  [c,c,c,c,c]]
        # kernel = [[-1.,-1.,-1.],
        #           [-1.,8,-1.],
        #           [-1.,-1.,-1.]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        plt.imshow(x1.data.numpy()[0],cmap='jet')
        # plt.show()
        plt.imshow(x2.data.numpy()[0],cmap='jet')
        # plt.show()
        plt.imshow(x3.data.numpy()[0],cmap='jet')
        # plt.show()
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=0)
        # plt.imshow(x1.data.numpy()[0,0,:,:],cmap='jet')
        # plt.show()
        # plt.imshow(x2.data.numpy()[0,0,:,:],cmap='jet')
        # plt.show()
        # plt.imshow(x3.data.numpy()[0,0,:,:],cmap='jet')
        # plt.show()
        x = torch.cat([x1, x2, x3], dim=1)
        # print(x)
        # outputs = []
        # x = self.conv1(x)
        return x

        

def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    cv.imshow("binary0", binary)

def get_feature(pic_dir):
    # 输入数据
    img = get_picture(pic_dir, transform)
    # 插入维度
    img = img.unsqueeze(0)
    img = img.to(device)

    # 特征输出
    net = FeatureExtractor().to(device)
    res = net(img).data.numpy()[0,0,:,:]
    
    plt.imshow(res)
    plt.show()
    r = []
    for i in res:
        a = []
        for j in i:
            if j < 0.35:
                a.append(-1)
            else:
                a.append(1)
        r.append(a)
    plt.imshow(r)   
    plt.show()

        
if __name__ == "__main__":
    
    get_feature('test_img/'+'img (2).jpg')