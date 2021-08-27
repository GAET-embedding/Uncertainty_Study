import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(256 * 2 * 2, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)
        self.sub_num = [1,2,3,4,5,6,7]


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def get_hidden(self, x):
        res = []
        num = len(x)
        x = self.maxpool(self.relu(self.conv1(x)))
        res.append(x.view([num, -1]).detach().cpu())

        x = self.maxpool(self.relu(self.conv2(x)))
        res.append(x.view([num, -1]).detach().cpu())

        x = self.relu(self.conv3(x))
        res.append(x.view([num, -1]).detach().cpu())

        x = self.relu(self.conv4(x))
        res.append(x.view([num, -1]).detach().cpu())

        x = self.maxpool(self.relu(self.conv5(x)))
        res.append(x.view([num, -1]).detach().cpu())

        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.relu(self.linear1(x))
        res.append(x.view([num, -1]).detach().cpu())

        x = self.relu(self.linear2(x))
        res.append(x.view([num, -1]).detach().cpu())
        return res