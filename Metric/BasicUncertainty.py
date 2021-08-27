import os
import numpy as np
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from BasicalClass import common_ten2numpy, common_predict
from BasicalClass import BasicModule


class BasicUncertainty(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, instance: BasicModule, device):
        super(BasicUncertainty, self).__init__()
        self.instance = instance
        self.device = device
        self.train_batch_size = instance.train_batch_size
        self.test_batch_size = instance.test_batch_size
        self.model = instance.model.to(device)
        self.class_num = instance.class_num
        self.save_dir = instance.save_dir
        self.module_id = instance.module_id
        self.softmax = nn.Softmax(dim=1)
        self.test_path = instance.test_path
        
        # handle train data and oracle
        self.train_y = instance.train_y
        self.train_pred_pos, self.train_pred_y =\
            instance.train_pred_pos, instance.train_pred_y
        self.train_loader = instance.train_loader
        self.train_num = len(self.train_y)
        self.train_oracle = np.int32(
            common_ten2numpy(self.train_pred_y).reshape([-1]) == \
                common_ten2numpy(self.train_y).reshape([-1])
        )
        # handle val data and oracle
        self.val_y = instance.val_y
        self.val_pred_pos, self.val_pred_y = \
            instance.val_pred_pos, instance.val_pred_y
        self.val_loader = instance.val_loader
        self.val_num = len(self.val_y)
        self.val_oracle = np.int32(
            common_ten2numpy(self.val_pred_y).reshape([-1]) == \
                common_ten2numpy(self.val_y).reshape([-1])
        )
        # handle ood data and oracle
        if instance.ood_path is not None:
            self.ood_y = instance.ood_y
            self.ood_pred_pos, self.ood_pred_y = \
                instance.ood_pred_pos, instance.ood_pred_y
            self.ood_loader = instance.ood_loader
            self.ood_num = len(self.ood_y)
            self.ood_oracle = np.int32(
                common_ten2numpy(self.ood_pred_y).reshape([-1]) == \
                    common_ten2numpy(self.ood_y).reshape([-1])
            )
        
        if self.test_path is not None:
            self.test_y = instance.test_y
            self.test_pred_pos, self.test_pred_y = \
                instance.test_pred_pos, instance.test_pred_y
            self.test_loader = instance.test_loader
            self.test_num = len(self.test_y)
            self.test_oracle = np.int32(
                common_ten2numpy(self.test_pred_y).reshape([-1]) == \
                    common_ten2numpy(self.test_y).reshape([-1])
            )
            
        else:
            self.test_y1, self.test_y2, self.test_y3 = \
                instance.test_y1, instance.test_y2, instance.test_y3
            self.test_pred_pos1, self.test_pred_y1 = \
                instance.test_pred_pos1, instance.test_pred_y1
            self.test_loader1 = instance.test_loader1
            self.test_num1 = len(self.test_y1)
            self.test_oracle1 = np.int32(
                common_ten2numpy(self.test_pred_y1).reshape([-1]) == \
                    common_ten2numpy(self.test_y1).reshape([-1])
            )
            self.test_pred_pos2, self.test_pred_y2 = \
                instance.test_pred_pos2, instance.test_pred_y2
            self.test_loader2 = instance.test_loader2
            self.test_num2 = len(self.test_y2)
            self.test_oracle2 = np.int32(
                common_ten2numpy(self.test_pred_y2).reshape([-1]) == \
                    common_ten2numpy(self.test_y2).reshape([-1])
            )
            self.test_pred_pos3, self.test_pred_y3 = \
                instance.test_pred_pos3, instance.test_pred_y3
            self.test_loader3 = instance.test_loader3
            self.test_num3 = len(self.test_y3)
            self.test_oracle3 = np.int32(
                common_ten2numpy(self.test_pred_y3).reshape([-1]) == \
                    common_ten2numpy(self.test_y3).reshape([-1])
            )
            

    @abstractmethod
    def _uncertainty_calculate(self, data_loader):
        return common_predict(data_loader, self.model, self.device, module_id=self.module_id)

    def run(self):
        score = self.get_uncertainty()
        self.save_uncertaity_file(score)
        print('finish score extract for class', self.__class__.__name__)
        return score

    def get_uncertainty(self):
        train_score = self._uncertainty_calculate(self.train_loader)
        val_score = self._uncertainty_calculate(self.val_loader)
        if self.instance.ood_path is not None:
            ood_score = self._uncertainty_calculate(self.ood_loader)
        if self.test_path is not None:
            test_score = self._uncertainty_calculate(self.test_loader)
            if self.instance.ood_path is not None:
                result = {
                    'train': train_score,
                    'val': val_score,
                    'test': test_score,
                    'ood': ood_score,
                }
            else:
                result = {
                    'train': train_score,
                    'val': val_score,
                    'test': test_score
                }
        else:
            test_score1 = self._uncertainty_calculate(self.test_loader1)
            test_score2 = self._uncertainty_calculate(self.test_loader2)
            test_score3 = self._uncertainty_calculate(self.test_loader3)
            if self.instance.ood_path is not None:
                result = {
                    'train': train_score,
                    'val': val_score,
                    'test1': test_score1,
                    'test2': test_score2,
                    'test3': test_score3,
                    'ood': ood_score
                }
            else:
                result = {
                    'train': train_score,
                    'val': val_score,
                    'test1': test_score1,
                    'test2': test_score2,
                    'test3': test_score3
                }
        return result

    def save_uncertaity_file(self, score_dict):
        data_name = self.instance.__class__.__name__
        uncertainty_type = self.__class__.__name__
        save_name = self.save_dir + '/' + data_name + '/' + uncertainty_type + '.res'
        if not os.path.isdir(os.path.join(self.save_dir, data_name)):
            os.mkdir(os.path.join(self.save_dir, data_name))
        torch.save(score_dict, save_name)
        print('get result for dataset %s, uncertainty type is %s' % (data_name, uncertainty_type))