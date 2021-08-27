import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

#
# class FeatureExtract(nn.Module):
#     def __init__(self, cfg, batch_norm=False):
#         super(FeatureExtract, self).__init__()
#         #self.layers = []
#         lay_index = 0
#         in_channels = 3
#         for v in cfg:
#             if v == 'M':
#                 #self.layers += [m]
#                 self.add_module(str(lay_index), nn.MaxPool2d(kernel_size=2, stride=2))
#                 lay_index += 1
#
#                 #self.layers += [m]
#                 self.add_module(str(lay_index), nn.Dropout(0.3))
#                 lay_index += 1
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#                 if batch_norm:
#                     self.add_module(str(lay_index), conv2d)
#                     lay_index += 1
#                     self.add_module(str(lay_index), nn.BatchNorm2d(v))
#                     lay_index += 1
#                     self.add_module(str(lay_index), nn.ReLU(inplace=True))
#                     lay_index += 1
#                 else:
#                     self.add_module(str(lay_index), conv2d)
#                     lay_index += 1
#                     self.add_module(str(lay_index), nn.ReLU(inplace=True))
#                     lay_index += 1
#                 self.add_module(str(lay_index), nn.Dropout(0.3))
#                 lay_index += 1
#                 in_channels = v
#
#     def get_hidden(self, x):
#         res = []
#         target_num = [44, 47, 50, 53]
#         data_size = len(x)
#         for i, layer in enumerate(self.modules()):
#             x = layer(x)
#             if i in target_num:
#                 res.append(x.view([data_size, -1]).detach().cpu())
#             if len(res) == len(target_num):
#                 return res
#         return res
#
#     def forward(self, x):
#         data_size = len(x)
#         res = x
#         for layer in self.children():
#             res = layer(res)
#         return res.view([data_size, -1])


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            #layers += [nn.Dropout(0.1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            #layers += [nn.Dropout(0.1)]
            in_channels = v
    return layers


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class VGG19(nn.Module):
    def add_features(self, features):
        lay_num = 0
        for layer in features:
            self.add_module(str(lay_num), layer)
            lay_num += 1
        f_layer = Flatten()
        self.add_module(str(lay_num), f_layer)

    def __init__(self, class_num=10):
        super(VGG19, self).__init__()
        self.sub_num = [30, 32, 34, 37, 39, 42]
        features = make_layers(cfg['E'])
        self.add_features(features)
        lay_num = len(features) + 1
        self.add_module(str(lay_num), nn.Dropout(0.3))
        lay_num += 1
        self.add_module(str(lay_num), nn.Linear(512, 512))
        lay_num += 1
        self.add_module(str(lay_num), nn.Dropout(0.3))
        lay_num += 1
        self.add_module(str(lay_num), nn.ReLU(True))
        lay_num += 1
        self.add_module(str(lay_num), nn.Linear(512, 512))
        lay_num += 1
        self.add_module(str(lay_num), nn.Dropout(0.3))
        lay_num += 1
        self.add_module(str(lay_num), nn.ReLU(True))
        lay_num += 1
        self.add_module(str(lay_num), nn.Linear(512, class_num))
        lay_num += 1

        for m in self.children():
           if isinstance(m, nn.Conv2d):
               n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
               m.weight.data.normal_(0, np.sqrt(2. / n))
               m.bias.data.zero_()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    def get_hidden(self, x):
        res = []
        data_num = len(x)
        for i, layer in enumerate(self.children()):
            x = layer(x)
            if i in self.sub_num:
                res.append(x.view([data_num, -1]).detach().cpu())
            if len(res) == len(self.sub_num):
                return res
        return res

    def get_feature(self, x):
        layer_num = len(list(self.children()))
        for i, layer in enumerate(self.children()):
            x = layer(x)
            if i == layer_num - 4:
                return x.detach().cpu()


if __name__ == '__main__':
    img = torch.zeros([1, 3, 32, 32])
    model = VGG19()
    y = model(img)
    h = model.get_hidden(img)
    print()