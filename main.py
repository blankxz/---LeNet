import os
import torch
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

ddd = '5'

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# plt.figure(dpi=150)
# Load training and testing datasets.

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


def get_picture_rgb(picture_dir):
    '''
    该函数实现了显示图片的RGB三通道颜色
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    # skimage.io.imsave('4.jpg',img256)

    # 取单一通道值显示
    for i in range(3):
        img = img256[:,:,i]
        ax = plt.subplot(1, 3, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        plt.imshow(img)
    plt.show()

    r = img256.copy()
    r[:,:,0:2]=0
    ax = plt.subplot(1, 4, 1)
    ax.set_title('B Channel')
    # ax.axis('off')
    plt.imshow(r)

    g = img256.copy()
    g[:,:,0]=0
    g[:,:,2]=0
    ax = plt.subplot(1, 4, 2)
    ax.set_title('G Channel')
    # ax.axis('off')
    plt.imshow(g)

    b = img256.copy()
    b[:,:,1:3]=0
    ax = plt.subplot(1, 4, 3)
    ax.set_title('R Channel')
    # ax.axis('off')
    plt.imshow(b)

    img = img256.copy()
    ax = plt.subplot(1, 4, 4)
    ax.set_title('image')
    # ax.axis('off')
    plt.imshow(img)

    # img = img256.copy()
    # ax = plt.subplot()
    # ax.set_title('image')
    # # ax.axis('off')
    # plt.imshow(img)

    plt.show()


def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb


class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        kernel = kernel.float()   # [channel_out, channel_in, kernel, kernel]

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out


class LeNet(nn.Module):
    '''
    该类继承了torch.nn.Modul类
    构建LeNet神经网络模型
    '''
    def __init__(self):
        super(LeNet, self).__init__()

        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        # a = torch.randn(20, 16, 50)
        # self.weight = nn.Parameter(torch.tensor([[1., 2., 3.], [4.,5.,6.],[1.,1.,1.]]), requires_grad=True)
        # self.sss = F.conv2d(a,self.weight)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 30, int(ddd), 1, 0, groups=1),   # input_size=(3*256*256)，padding=2
            nn.ReLU(),                  # input_size=(32*256*256)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(32*128*128)
        )

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

    # 定义前向传播过程，输入为x
    def forward(self, x):
        print('hello')
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
step = 1
# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        print(self.submodule._modules.items())
        try:
            for name, module in self.submodule._modules.items():
                if "fc" in name: 
                    print(name)
                    x = x.view(x.size(0), -1)
                global step
                print(step)
                step += 1
                print(module)
                x = module(x)
                print(name)
                if name in self.extracted_layers:
                    outputs.append(x)
                print(x)
        except:
            return outputs


def get_feature(pic_dir):
    # 输入数据
    img = get_picture(pic_dir, transform)
    # 插入维度
    img = img.unsqueeze(0)
    img = img.to(device)

    # 特征输出
    net = LeNet().to(device)
    # net.load_state_dict(torch.load('./model/net_050.pth'))
    exact_list = ["conv1","conv2"]
    # exact_list = ["conv1"]
    myexactor = FeatureExtractor(net, exact_list)
    x = myexactor(img)
    print('=========================')
    # 特征输出可视化
    for i in range(30):
        ax = plt.subplot(5, 6, i + 1)
        # ax.set_title('Feature {}'.format(i),fontsize=5)
        ax.axis('off')
        plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
    plt.savefig('res'+ddd+'_img/'+pic_dir.split('/')[1])
    # plt.show()

# 训练
if __name__ == "__main__":
    pic_dir = os.listdir('test_img')
    try:
        for i in range(1,8,2):
            os.mkdir('res'+str(i)+'_img')
    except:
        pass
    for i in pic_dir:
    # get_picture_rgb(pic_dir)
        get_feature('test_img/'+i)
    
