import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip, struct
import numpy as np
import torch.utils.data as data
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision


class GaussConv2d(nn.Module):
    def __init__(self):
        super(GaussConv2d, self).__init__()
        gauss = [[0.03797616, 0.044863533, 0.03797616],
         [0.044863533, 0.053, 0.044863533],
         [0.03797616, 0.044863533, 0.03797616]]
        gauss_weight = torch.FloatTensor(gauss).unsqueeze(0).unsqueeze(0)
        self.gauss_weight = nn.Parameter(data=gauss_weight, requires_grad=False)
    
    def forward(self, x):
        gauss_res = F.conv_transpose2d(x, self.gauss_weight, padding=0)
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
        profile_res = F.conv2d(x, self.profile_weight, padding=2)
        return profile_res


# LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

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
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(49,1500),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(1500,1000),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(1000,10)

    def forward(self, x):

        x = self.conv1(x)   # conv1 -> relu -> pooling -> conv2 -> pooling
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Mnist数据集
class MnistDataset(data.Dataset):
    def __init__(self, path, train=True):
        self.path = path
        if train:
            X, y = self._read('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
        else:
            X, y = self._read('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

        self.images = torch.from_numpy(X.reshape(-1, 1, 28, 28))
        self.labels = torch.from_numpy(y.astype(int))

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

    def _read(self, image, label):
        with gzip.open(self.path + image, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(-1, rows, cols)
        with gzip.open(self.path + label) as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y = np.frombuffer(flbl.read(), dtype=np.int8)
        return X, y


def imshow(image):
    '''
    图片展示
    :param image:
    :return:
    '''
    npimg = image.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


train_dataset = MnistDataset('./datas/')
train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=60, num_workers=0)

# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))

net = LeNet()   # LeNet网络
criterion = nn.CrossEntropyLoss()   # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # 带动量的随机梯度下降

# 模型训练
print('------------Start Training------------')
for epoch in range(3):      # epoch为3
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data   # 输入
        inputs = inputs.float()
        labels = labels.long()
        # inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()   # 梯度清零
        outputs = net(inputs)
        loss = criterion(outputs, labels)   # 损失
        loss.backward()     # 反向传播
        optimizer.step()    # 梯度更新

        running_loss += loss.item()
        if i % 100 == 99:
            print('epoch: %d, count: %5d, loss: %.3f' % (epoch + 1, (i + 1)*60, running_loss / 100))
            running_loss = 0.0
print('------------Finish Training------------\n')

# 准确率测定
test_dataset = MnistDataset('./datas/')
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=10, num_workers=0)

corrent = 0     # 正确数
total = 0   # 总数
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.float()
        test_labels = test_labels.long()
        outputs = net(test_images)
        _, predicted = torch.max(outputs, dim=1)
        c = (predicted == test_labels).squeeze()
        total += test_labels.size(0)
        corrent += (predicted == test_labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * corrent / total))
