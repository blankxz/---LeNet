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

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    im = Image.open(picture_dir)
    im = im.resize((256, 256))
    Lim = im.convert('L' )
    Lim.show()
    return transform(Lim)

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
        k1 = [[0.03797616, 0.044863533, 0.03797616],
         [0.044863533, 0.053, 0.044863533],
         [0.03797616, 0.044863533, 0.03797616]]
        kk1 = torch.FloatTensor(k1).unsqueeze(0).unsqueeze(0)
        k = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight1 = nn.Parameter(data=kk1, requires_grad=False)
        self.weight = nn.Parameter(data=k, requires_grad=False)
        x1 = x[:, 0]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight1, padding=0)
        plt.imshow(x1.data.numpy()[0,0,:,:],cmap='jet')
        plt.show()
        x0 = F.conv2d(x1, self.weight, padding=0)

        return x0


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
    
        
    d = np.array(res)
    W2, H2 = d.shape
    dx = np.zeros([W2-1, H2-1])
    dy = np.zeros([W2-1, H2-1])
    NMS = np.copy(d)
    NMS[0,:] = NMS[W2-1,:] = NMS[:,0] = NMS[:, H2-1] = 0
    for i in range(1, W2-1):
        for j in range(1, H2-1):

            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]
                
                # 如果Y方向幅度值较大
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i-1, j]
                    grad4 = d[i+1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i-1, j+1]
                        grad3 = d[i+1, j-1]
                        
                # 如果X方向幅度值较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j-1]
                    grad4 = d[i, j+1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i+1, j-1]
                        grad3 = d[i-1, j+1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
            
                gradTemp1 = weight * grad1 + (1-weight) * grad2
                gradTemp2 = weight * grad3 + (1-weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0
    # plt.imshow(d)   
    # plt.show()

    r = []
    for i in d:
        a = []
        for j in i:
            if j < 0.1:
                a.append(-1)
            else:
                a.append(1)
        r.append(a)
    plt.imshow(r)   
    plt.show()
        
if __name__ == "__main__":
    
    get_feature('test_img/'+'img (7).jpg')