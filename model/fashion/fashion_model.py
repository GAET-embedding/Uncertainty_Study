from __future__ import print_function   # 从future版本导入print函数功能
import torch.nn as nn                   # 指定torch.nn别名nn
import torch.nn.functional as F         # 引用神经网络常用函数包，不具有可学习的参数
import numpy as np


class Fashion_MLP(nn.Module):
    def __init__(self):
        super(Fashion_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 500) # 784表示输入神经元数量，1000表示输出神经元数量
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
        self.dropout = nn.Dropout()
        self.name = 'Fashion_MLP'
        self.sub_num = [1, 2]

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_hidden(self, x):
        res = []
        num = len(x)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        res.append(x.view([num, -1]).detach().cpu())

        x = F.relu(self.fc2(x))
        res.append(x.view([num, -1]).detach().cpu())
        return res


class Fashion_CNN(nn.Module):
    def __init__(self):
        super(Fashion_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 200)
        self.fc2 = nn.Linear(200, 10)
        self.dropout = nn.Dropout()
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_3 = nn.ReLU(inplace=True)
        self.name = 'Fashion_CNN'
        self.sub_num = [2, 4, 5]

    def try_relu(self, func, x):
        try:
            x = func(x)
        except:
            x = x
        return x

    def forward(self, x):
        x = self.conv1(x)
        self.try_relu(self.relu_1, x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        self.try_relu(self.relu_2, x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.dropout(x)
        x = self.fc1(x)
        self.try_relu(self.relu_3, x)
        x = self.fc2(x)
        return x

    def get_hidden(self, x):
        res = []
        num = len(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        res.append(x.view([num, -1]).detach().cpu())

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        res.append(x.view([num, -1]).detach().cpu())

        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        res.append(x.view([num, -1]).detach().cpu())

        return res

    def get_activation(self, x):
        res = []
        x = F.relu(self.conv1(x))
        res.append(x.detach().cpu().numpy())

        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        res.append(x.detach().cpu().numpy())

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        res.append(x.detach().cpu().numpy())
        return np.concatenate(res, axis=1)

    def get_feature(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.dropout(x)
        x = self.fc1(x)
        return x.detach().cpu()