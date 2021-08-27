import os
import torch
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader, Subset
from BasicalClass.common_function import common_predict, common_ten2numpy


class BasicModule:
    __metaclass__ = ABCMeta

    def __init__(self, device, res_dir, save_dir, data_dir, ood_dir, module_id, 
                 train_batch_size, test_batch_size, max_size, load_poor=False):

        if module_id == 0: # code summary
            self.tk_path = os.path.join(data_dir, 'tk.pkl')
            self.train_path = os.path.join(data_dir, 'train.pkl')
            test_path = os.path.join(data_dir, 'test.pkl')
            if ood_dir is None:
                self.ood_path = None
            else:
                self.ood_path = os.path.join(ood_dir, 'test.pkl')

            if os.path.exists(test_path):
                self.test_path = test_path
                self.test_path1 = None
                self.test_path2 = None
                self.test_path3 = None
            else:
                self.test_path = None
                self.test_path1 = os.path.join(data_dir, 'test1.pkl')
                self.test_path2 = os.path.join(data_dir, 'test2.pkl')
                self.test_path3 = os.path.join(data_dir, 'test3.pkl')
                
            self.val_path = os.path.join(data_dir, 'val.pkl')

        elif module_id == 1: # code completion
            self.tk_path = None
            self.train_path = os.path.join(data_dir, 'train.tsv')
            test_path = os.path.join(data_dir, 'test.tsv')
            self.ood_path = os.path.join(ood_dir, 'val.tsv')

            if os.path.exists(test_path):
                self.test_path = test_path
                self.test_path1 = None
                self.test_path2 = None
                self.test_path3 = None
            else:
                self.test_path = None
                self.test_path1 = os.path.join(data_dir, 'test1.tsv')
                self.test_path2 = os.path.join(data_dir, 'test2.tsv')
                self.test_path3 = os.path.join(data_dir, 'test3.tsv')
            
            self.val_path = os.path.join(data_dir, 'val.tsv')
            self.min_samples = 5
        else:
            raise TypeError()
        
        self.module_id = module_id
        self.vec_path = 'java_dataset/embedding_vec/100_2/Doc2VecEmbedding0.vec'
        self.embed_dim = 120
        self.res_dir = res_dir
        self.max_size = max_size # use part of the data
        self.embed_type = 1
        self.device = device
        self.load_poor = load_poor
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.model = self.get_model()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.test_loader1 = None
        self.test_loader2 = None
        self.test_loader3 = None
        self.ood_loader=None
        self.save_dir = save_dir
        

        if not os.path.exists(save_dir):
            if '/' in save_dir: # iteratively make dir
                os.makedirs(save_dir)
            else:   
                os.mkdir(save_dir)
        if not os.path.isdir(
            os.path.join(save_dir, self.__class__.__name__)
        ):
            os.mkdir(os.path.join(save_dir, self.__class__.__name__))

    def get_model(self):
        if not self.load_poor:
            model = self.load_model()
        else:
            model = self.load_poor_model()
        model.to(self.device)
        model.eval()
        print('model name is ', model.__class__.__name__)
        return model

    @abstractmethod
    def load_model(self):
        return None

    @abstractmethod
    def load_poor_model(self):
        return None

    def get_hiddenstate(self, dataloader, device):
        sub_num = self.model.sub_num
        hidden_res, label_res = [[] for _ in sub_num], []

        if self.module_id == 0: # code summary
            for (sts, paths, eds), y, length in dataloader:
                sts = sts.to(device)
                paths = paths.to(device)
                eds = eds.to(device)
                res = self.model.get_hidden(sts, paths, eds, length)
                y = torch.tensor(y, dtype=torch.long) # convert tuple to tensor
                # detach
                sts = sts.detach().cpu()
                paths = paths.detach().cpu()
                eds = eds.detach().cpu()
                res = [s.detach().cpu() for s in res]

                for i, r in enumerate(res):
                    hidden_res[i].append(r)
                label_res.append(y)

        elif self.module_id == 1: # code completion
            for input, y, _ in dataloader:
                input = input.to(device)
                res = self.model.get_hidden(input)

                # detach
                input = input.detach().cpu()
                res = [s.detach().cpu() for s in res]
            
                for i, r in enumerate(res):
                    hidden_res[i].append(r)
                label_res.append(y.long())

        else:
            raise TypeError()

        hidden_res = [torch.cat(tmp, dim=0) for tmp in hidden_res]
        return hidden_res, sub_num, torch.cat(label_res)


    def get_information(self):
        self.train_pred_pos, self.train_pred_y, self.train_y = \
            common_predict(self.train_loader, self.model, self.device, module_id=self.module_id)

        self.val_pred_pos, self.val_pred_y, self.val_y = \
            common_predict(self.val_loader, self.model, self.device, module_id=self.module_id)

        if self.ood_loader is not None:
            self.ood_pred_pos, self.ood_pred_y, self.ood_y = \
                common_predict(self.ood_loader, self.model, self.device, module_id=self.module_id)
        else:
            self.ood_pred_pos, self.ood_pred_y, self.ood_y = None, None, None

        if self.test_path is not None: # only one test set
            self.test_pred_pos, self.test_pred_y, self.test_y = \
                common_predict(self.test_loader, self.model, self.device, module_id=self.module_id)
            print(
                'train class num: {}, val class num: {}, test class num: {}, ood class num: {}'.format(
                    self.train_pred_pos.size(1), 
                    self.val_pred_pos.size(1), 
                    self.test_pred_pos.size(1),
                    self.train_pred_pos.size(1),
                ))
        else:
            self.test_pred_pos1, self.test_pred_y1, self.test_y1 = \
                common_predict(self.test_loader1, self.model, self.device, module_id=self.module_id)
            self.test_pred_pos2, self.test_pred_y2, self.test_y2 = \
                common_predict(self.test_loader2, self.model, self.device, module_id=self.module_id)
            self.test_pred_pos3, self.test_pred_y3, self.test_y3 = \
                common_predict(self.test_loader3, self.model, self.device, module_id=self.module_id)
            print(
                'train class num: {}, val class num: {}, test1 class num: {}, test2 class num: {}, test3 class num: {}, ood class num: {}'.format(
                    self.train_pred_pos.size(1), 
                    self.val_pred_pos.size(1), 
                    self.test_pred_pos1.size(1),
                    self.test_pred_pos2.size(1),
                    self.test_pred_pos3.size(1),
                    self.train_pred_pos.size(1),
                ))

        self.class_num = self.train_pred_pos.size(1) # setting the class_num
        

    def save_truth(self):
        self.train_truth = self.train_pred_y.eq(self.train_y)
        self.val_truth = self.val_pred_y.eq(self.val_y)
        if self.ood_pred_y is not None and self.ood_y is not None:
            self.ood_truth = self.ood_pred_y.eq(self.ood_y)
        else:
            self.ood_truth = None
        if self.test_path is not None:
            self.test_truth = self.test_pred_y.eq(self.test_y)
            if self.ood_truth is not None:
                truth = {
                    'train': common_ten2numpy(self.train_truth), # torch to numpy cpu
                    'val': common_ten2numpy(self.val_truth),
                    'test': common_ten2numpy(self.test_truth),
                    'ood': common_ten2numpy(self.ood_truth)
                }
            else:
                truth = {
                    'train': common_ten2numpy(self.train_truth), # torch to numpy cpu
                    'val': common_ten2numpy(self.val_truth),
                    'test': common_ten2numpy(self.test_truth)
                }
        else:
            self.test1_truth = self.test_pred_y1.eq(self.test_y1)
            self.test2_truth = self.test_pred_y2.eq(self.test_y2)
            self.test3_truth = self.test_pred_y3.eq(self.test_y3)
            if self.ood_truth is not None:
                truth = {
                    'train': common_ten2numpy(self.train_truth), # torch to numpy cpu
                    'val': common_ten2numpy(self.val_truth),
                    'test1': common_ten2numpy(self.test1_truth),
                    'test2': common_ten2numpy(self.test2_truth),
                    'test3': common_ten2numpy(self.test3_truth),
                    'ood': common_ten2numpy(self.ood_truth)
                }
            else:
                    truth = {
                    'train': common_ten2numpy(self.train_truth), # torch to numpy cpu
                    'val': common_ten2numpy(self.val_truth),
                    'test1': common_ten2numpy(self.test1_truth),
                    'test2': common_ten2numpy(self.test2_truth),
                    'test3': common_ten2numpy(self.test3_truth)
                }
                
        torch.save(
            truth, 
            os.path.join(self.save_dir, self.__class__.__name__) + '/truth.res'
        )