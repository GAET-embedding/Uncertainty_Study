  
from model.fashion import *
import torchvision
from torchvision import transforms
import torch.nn as nn
from BasicalClass.common_function import *
from BasicalClass.BasicModule import BasicModule
import torch.optim as optim


class AgNews_Module(BasicModule):
    def __init__(self, device, load_poor=False):
        super(AgNews_Module, self).__init__(device, load_poor)
        self.train_batch_size = 512
        self.test_batch_size = 512 if IS_DEBUG else 5000
        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        self.get_information()
        self.input_shape = (300, )
        self.class_num = 4

        self.test_acc = common_cal_accuracy(self.test_pred_y, self.test_y)
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)

        self.save_truth()
        print('construct the module', self.__class__.__name__, 'the accuracy is %0.3f, %0.3f' % (self.train_acc, self.test_acc))

    def load_model(self):
        model = Fashion_CNN()
        model.load_state_dict(
            torch.load('../model_weight/fashion/' + model.name + '.h5', map_location=self.device)
        )
        return model

    def load_poor_model(self):
        model = Fashion_MLP()
        model.load_state_dict(
            torch.load('../model_weight/fashion/' + model.name + '.h5', map_location=self.device)
        )
        return model

    def load_data(self):
        train_db = torch.load('../data/AG_NEWS' + '_train.pt')
        val_db = torch.load('../data/AG_NEWS' + '_val.pt')
        test_db = torch.load('../data/AG_NEWS' + '_test.pt')
        return self.get_loader(train_db, val_db, test_db)


if __name__ == '__main__':
    train_db = torch.load('../data/AG_NEWS' + '_train.pt')
    val_db = torch.load('../data/AG_NEWS' + '_val.pt')
    test_db = torch.load('../data/AG_NEWS' + '_test.pt')
    print()