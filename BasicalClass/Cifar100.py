  
from model.cifar_100 import *
import torchvision
from torchvision import transforms
import torch.nn as nn
from BasicalClass.common_function import *
from BasicalClass.BasicModule import BasicModule
import torch.optim as optim


class CIFAR100_Module:
    def __init__(self, device, load_poor = False):
        self.mean = [0.507, 0.487, 0.441] ####Todo  Modeity this
        self.std = [0.267, 0.256, 0.276]  ####Todo  Modeity this
        min_val = (0 - np.array(self.mean)) / np.array(self.std)
        max_val = (1 - np.array(self.mean)) / np.array(self.std)
        self.clip = (min(min_val), max(max_val))
        self.device = device
        self.load_poor = load_poor
        if not load_poor:
            self.model = self.load_model().to(self.device)
        else:
            self.model = self.load_poor_model().to(self.device)
        self.model.eval()
        print('model name is ', self.model.__class__.__name__)

        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )
        self.train_batch_size = 32
        self.test_batch_size = 32 if IS_DEBUG else 50
        self.name = 'CIFAR100'
        self.loss = nn.CrossEntropyLoss()
        train_dataset = self.load_data(True)
        self.train_x, self.train_y = common_get_xy(train_dataset, self.test_batch_size, self.device)
        self.train_pred_pos, self.train_pred_y = \
            common_predict(self.train_x, self.model, self.test_batch_size, self.device)


        test_dataset = self.load_data(False)
        self.test_x, self.test_y = common_get_xy(test_dataset, self.test_batch_size, self.device)
        self.test_pred_pos, self.test_pred_y = \
            common_predict(self.test_x, self.model, self.test_batch_size, self.device)


        self.input_shape = (3, 32, 32)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer =  optim.Adam(self.model.parameters(), lr=0.01)
        self.class_num = 100
        self.acc = common_cal_accuracy(self.test_pred_y, self.test_y)
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)
        self.eps = 0.3
        if not os.path.isdir('./' + self.name):
            os.mkdir('./' + self.name)

        print('construct the module', self.name, 'the accuracy is %0.3f, %0.3f' % (self.train_acc, self.acc))
        print('training data number is', len(self.train_x), 'test data number is ', len(self.test_x))

    def load_model(self):
        model = resneXt_cifar(depth=29, cardinality=8, baseWidth=64, num_classes=100)
        model = torch.nn.DataParallel(model, device_ids=[self.device])
        state_dict = torch.load('./model_weight/cifar_100/resnext.pth.tar', map_location=self.device)
        model.load_state_dict(state_dict['state_dict'])
        return model.module

    def load_poor_model(self):
        model = SqueezeNet()
        state_dict = torch.load('./model_weight/cifar_100/SqueezeNet.h5', map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def load_data(self, is_train=True):
        return torchvision.datasets.CIFAR100(
            'data/cifar_100', train=is_train, transform=self.transform_train, target_transform=None, download=True)

    def get_hiddenstate(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.train_batch_size,
            shuffle=False, collate_fn=None,
        )
        data_num = 0
        sub_num = self.model.sub_num
        sub_res_list = [[] for _ in sub_num]
        for i, x in enumerate(data_loader):
            data_num += len(x)
            x = x.to(self.device)
            self.model.to(self.device)
            sub_y = self.model.get_hidden(x)
            for j in range(len(sub_num)):
                sub_res_list[j].append(sub_y[j])
            if IS_DEBUG and i >= DEBUG_NUM:
                break
        sub_res_list = [torch.cat(i, dim = 0) for i in sub_res_list]
        return sub_res_list, sub_num