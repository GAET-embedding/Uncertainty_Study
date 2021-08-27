from model.cifar_10.vgg19 import VGG19
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from BasicalClass.common_function import *
from BasicalClass.BasicModule import BasicModule


class CIFAR10_Module(BasicModule):
    def __init__(self, device, load_poor=False):
        super(CIFAR10_Module, self).__init__(device, load_poor)
        self.train_batch_size = 256
        self.test_batch_size = 256 if IS_DEBUG else 2000
        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        self.get_information()
        self.input_shape = (3, 32, 32)
        self.class_num = 10

        self.test_acc = common_cal_accuracy(self.test_pred_y, self.test_y)
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)

        self.save_truth()
        print('construct the module', self.__class__.__name__,
              'the accuracy is %0.3f, %0.3f' % (self.train_acc, self.test_acc))

    def load_model(self):
        model = VGG19()
        model.load_state_dict(
            torch.load('../model_weight/cifar10/VGG19.h5', map_location=self.device)
        )
        return model

    def load_poor_model(self):
        return None  #todo
        #model = AlexNet()
        ##state_dict = torch.load('../model_weight/cifar_10/AlexNet.h5', map_location=self.device)
        #model.load_state_dict(state_dict)
        #return model

    def load_data(self):
        train_db = torch.load('../data/cifar10' + '_train.pt')
        val_db = torch.load('../data/cifar10' + '_val.pt')
        test_db = torch.load('../data/cifar10' + '_test.pt')
        return self.get_loader(train_db, val_db, test_db)

    # def get_hiddenstate(self, dataset):
    #     data_loader = DataLoader(dataset, batch_size=self.train_batch_size,
    #         shuffle=False, collate_fn=None,
    #     )
    #     data_num = 0
    #     sub_num = self.model.sub_num
    #     sub_res_list = [[] for _ in sub_num]
    #     for i, x in enumerate(data_loader):
    #         data_num += len(x)
    #         x = x.to(self.device)
    #         self.model.to(self.device)
    #         sub_y = self.model.get_hidden(x)
    #         for j in range(len(sub_num)):
    #             sub_res_list[j].append(sub_y[j])
    #         if IS_DEBUG and i >= DEBUG_NUM:
    #             break
    #     sub_res_list = [torch.cat(i, dim=0) for i in sub_res_list]
    #     return sub_res_list, sub_num
    #