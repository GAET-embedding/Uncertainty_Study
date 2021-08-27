import os
import torch.nn as nn
import torch.optim as optim
from BasicalClass.common_function import *
from BasicalClass.BasicModule import BasicModule
from program_tasks.code_summary.Code2VecModule import Code2Vec
from program_tasks.code_summary.CodeLoader import CodeLoader
from program_tasks.code_summary.main import perpare_train, my_collate
from preprocess.checkpoint import Checkpoint


class CodeSummary_Module(BasicModule):

    def __init__(self, device, res_dir, save_dir, data_dir, ood_dir,
                 module_id, train_batch_size, test_batch_size, max_size, load_poor=False):
        super(CodeSummary_Module, self).__init__(
            device, res_dir, save_dir, data_dir, ood_dir, module_id,
            train_batch_size, test_batch_size, max_size, load_poor
        )

        if self.test_path is not None: # only on test test
            self.train_loader, self.val_loader, self.test_loader, self.ood_loader = self.load_data()
        else:
            self.train_loader, self.val_loader, self.test_loader1, \
                self.test_loader2, self.test_loader3, self.ood_loader = self.load_data()

        self.get_information()
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)
        self.val_acc = common_cal_accuracy(self.val_pred_y, self.val_y)
        if self.ood_path is not None:
            self.ood_acc = common_cal_accuracy(self.ood_pred_y, self.ood_y)

        if self.test_path is not None:
            self.test_acc = common_cal_accuracy(self.test_pred_y, self.test_y)
            self.save_truth()
            if self.ood_path is not None:
                print(
                    'construct the module {}: '.format(self.__class__.__name__), 
                    'train acc %0.4f, val acc %0.4f, test acc %0.4f, ood acc %0.4f' % (
                        self.train_acc, self.val_acc, self.test_acc, self.ood_acc)
                )
            else:
                print(
                    'construct the module {}: '.format(self.__class__.__name__), 
                    'train acc %0.4f, val acc %0.4f, test acc %0.4f' % (
                        self.train_acc, self.val_acc, self.test_acc)
                )
        else:
            self.test_acc1 = common_cal_accuracy(self.test_pred_y1, self.test_y1)
            self.test_acc2 = common_cal_accuracy(self.test_pred_y2, self.test_y2)
            self.test_acc3 = common_cal_accuracy(self.test_pred_y3, self.test_y3)
            self.save_truth()
            if self.ood_path is not None:
                print(
                    'construct the module {}: '.format(self.__class__.__name__), 
                    'train acc %0.4f, val acc %0.4f, test1 acc %0.4f, test2 acc %0.4f, test3 acc %0.4f, ood acc %0.4f' % (
                        self.train_acc, self.val_acc, self.test_acc1, self.test_acc2, self.test_acc3, self.ood_acc)
                )
            else:
                print(
                    'construct the module {}: '.format(self.__class__.__name__), 
                    'train acc %0.4f, val acc %0.4f, test1 acc %0.4f, test2 acc %0.4f, test3 acc %0.4f' % (
                        self.train_acc, self.val_acc, self.test_acc1, self.test_acc2, self.test_acc3)
                )


    def load_model(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model

        return model


    def load_poor_model(self):
        oldest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(oldest_checkpoint_path)
        model = resume_checkpoint.model
        return model

    def load_data(self):
        token2index, path2index, func2index, embed, tk2num = perpare_train(
            self.tk_path, self.embed_type, self.vec_path, self.embed_dim, self.res_dir
        )
        train_db = CodeLoader(self.train_path, self.max_size, token2index, tk2num)
        val_db = CodeLoader(self.val_path, self.max_size, token2index, tk2num)
        train_loader = DataLoader(
            train_db, batch_size=self.train_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        val_loader = DataLoader(
            val_db, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        ood_loader = None
        if self.ood_path is not None:
            # set the OOD data size the same as validation set
            if self.max_size is not None:
                ood_size = min(self.max_size, len(val_db))
            else:
                ood_size = len(val_db)

            ood_db = CodeLoader(
                self.ood_path, ood_size, token2index, tk2num, 
                path2index=path2index, func2index=func2index
            )
            ood_loader = DataLoader(
                ood_db, batch_size=self.train_batch_size, 
                collate_fn=my_collate, shuffle=False
            )

        if self.test_path is not None:
            test_db = CodeLoader(self.test_path, self.max_size, token2index, tk2num)
            if self.ood_path is not None:
                print('train data {}, val data {}, test data {}, ood data {}'.format(
                    len(train_db), len(val_db), len(test_db), len((ood_db))
                ))
            else:
                print('train data {}, val data {}, test data {}'.format(
                    len(train_db), len(val_db), len(test_db)
                ))
            test_loader = DataLoader(
                test_db, batch_size=self.test_batch_size, 
                collate_fn=my_collate, shuffle=False
            )
            if self.ood_path is not None:
                print('train loader {}, val loader {}, test loader {}, ood loader {}'.format(
                    len(train_loader), len(val_loader), len(test_loader), len(ood_loader)
                ))
            else:
                print('train loader {}, val loader {}, test loader {}'.format(
                    len(train_loader), len(val_loader), len(test_loader)
                ))

            return train_loader, val_loader, test_loader, ood_loader

        else:
            test_db1 = CodeLoader(self.test_path1, self.max_size, token2index, tk2num)
            test_db2 = CodeLoader(self.test_path2, self.max_size, token2index, tk2num)
            test_db3 = CodeLoader(self.test_path3, self.max_size, token2index, tk2num)
            if self.ood_path is not None:
                print('train data {}, val data {}, test data1 {}, test data2 {}, test data3 {}, ood data {}'.format(
                    len(train_db), len(val_db), len(test_db1), len(test_db2), len(test_db3), len(ood_db)
                ))
            else:
                print('train data {}, val data {}, test data1 {}, test data2 {}, test data3 {}'.format(
                    len(train_db), len(val_db), len(test_db1), len(test_db2), len(test_db3)
                ))
            test_loader1 = DataLoader(
                test_db1, batch_size=self.test_batch_size, 
                collate_fn=my_collate, shuffle=False
            )
            test_loader2 = DataLoader(
                test_db2, batch_size=self.test_batch_size, 
                collate_fn=my_collate, shuffle=False
            )
            test_loader3 = DataLoader(
                test_db3, batch_size=self.test_batch_size, 
                collate_fn=my_collate, shuffle=False
            )
            if self.ood_path is not None:
                print('train loader {}, val loader {}, test loader1 {}, test loader2 {}, test loader3 {}, ood loader {}'.format(
                    len(train_loader), len(val_loader), len(test_loader1), len(test_loader2), len(test_loader3), len(ood_loader)
                ))
            else:
                print('train loader {}, val loader {}, test loader1 {}, test loader2 {}, test loader3 {}'.format(
                    len(train_loader), len(val_loader), len(test_loader1), len(test_loader2), len(test_loader3)
                ))
            return train_loader, val_loader, test_loader1, test_loader2, test_loader3, ood_loader
        

if __name__ == '__main__':
    CodeSummary_Module(DEVICE)