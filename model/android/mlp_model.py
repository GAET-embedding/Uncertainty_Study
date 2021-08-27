import torch.nn as nn
import numpy as np


class Android_MLP(nn.Module):
    def __init__(self, feature_num, drop_p = 0.5):
        super(Android_MLP, self).__init__()
        self.feature_num = feature_num
        self.drop_p = drop_p
        self.linear_1 = nn.Linear(feature_num, 100)
        self.linear_2 = nn.Linear(100, 100)
        self.linear_3 = nn.Linear(100, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_p)
        self.sub_num = [1, 2]

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_3(x)
        return x

    def get_hidden(self, x):
        res = []
        num = len(x)
        x = self.linear_1(x)
        res.append(x.view([num, -1]).detach().cpu())
        x = self.relu(x)
        x = self.linear_2(x)
        res.append(x.view([num, -1]).detach().cpu())
        return res

    def get_activation(self, x):
        res = []
        x = self.linear_1(x)
        x = self.relu(x)
        res.append(x.detach().cpu().numpy())

        x = self.linear_2(x)
        x = self.relu(x)
        res.append(x.detach().cpu().numpy())
        return np.concatenate(res, axis= 1)


class Android_Poor(nn.Module):
    def __init__(self, feature_num, drop_p = 0.5):
        super(Android_Poor, self).__init__()
        self.feature_num = feature_num
        self.drop_p = drop_p
        self.linear_1 = nn.Linear(feature_num, 50)
        self.linear_2 = nn.Linear(50, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_p)
        self.sub_num = [1]

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

    def get_hidden(self, x):
        res = []
        num = len(x)
        x = self.linear_1(x)
        res.append(x.view([num, -1]).detach().cpu())
        return res

    def get_activation(self, x):
        res = []
        x = self.linear_1(x)
        x = self.relu(x)
        res.append(x.detach().cpu().numpy())
        return np.concatenate(res, axis= 1)