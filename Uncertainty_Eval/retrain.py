from operator import index
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import datetime
from Metric import *
from BasicalClass import CodeSummary_Module
from program_tasks.code_summary.main import train_model, test_model, perpare_train, my_collate, dict2list
from program_tasks.code_summary.CodeLoader import CodeLoader
from preprocess.checkpoint import Checkpoint


# ModuleList = [
#     # Fashion_Module,
#     # CIFAR10_Module,
#     CodeSummary_Module
# ]
# MetricList = [
#     Vanilla,
#     ModelWithTemperature,
#     PVScore,
#     # Mahalanobis,
#     ModelActivateDropout,
#     Entropy,
#     Mutation,
# ]


class BasicRetrainModule:

    MetricNames = [
        'Entropy.res',
        'ModelActiveDropout.res',
        'ModelWithTemperature.res',
        'Mutation.res',
        'PVScore.res',
        'Vanilla.res',
    ]
    def __init__(self, module_id, metric_id, device, res_dir, data_dir, 
                 metric_dir, save_dir, train_batch, epochs, embed_dim=100):
        self.res_dir = res_dir
        self.save_dir = save_dir
        self.device = device
        self.tk_path = os.path.join(data_dir, 'tk.pkl')
        self.train_data = os.path.join(data_dir, 'train.pkl')
        self.val_data = os.path.join(data_dir, 'val.pkl')
        self.shift1_data = os.path.join(data_dir, 'test1.pkl')
        self.shift2_data = os.path.join(data_dir, 'test2.pkl')
        self.shift3_data = os.path.join(data_dir, 'test3.pkl')
        self.embed_type = 1
        self.embed_dim = embed_dim
        self.vec_path = None
        self.train_batch = train_batch
        self.epochs = epochs
        self.metricname = self.MetricNames[metric_id][:-4]
        
        self.model, self.optimizer, self.start_epoch = self.load_model()
        metric_res, mU_shift1, mU_shift2, mU_shift3 = self.load_metric(
            metric_id, metric_dir
        )
   
        self.val_loader, self.test_trainloader1, self.test_valloader1, \
            self.test_trainloader2, self.test_valloader2, self.test_trainloader3, \
                self.test_valloader3, self.index2func = self.load_data(
                mU_shift1, mU_shift2, mU_shift3, metric_res
            )
        
        self.criterian = nn.CrossEntropyLoss()  # loss
        print('before retraining, model performance: ')
        test_model(self.val_loader, self.model, self.device, self.index2func, 'val')
        test_model(self.test_valloader1, self.model, self.device, self.index2func, 'shift1')
        test_model(self.test_valloader2, self.model, self.device, self.index2func, 'shift2')
        test_model(self.test_valloader3, self.model, self.device, self.index2func, 'shift3')



    def load_model(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        start_epoch = resume_checkpoint.epoch
        model.to(self.device)
        model.eval()
        print('model name is ', model.__class__.__name__)
        return model, optimizer, start_epoch

    def load_metric(self, metric_id, metric_dir):
        metric_res = torch.load(os.path.join(metric_dir, self.MetricNames[metric_id]))
        # mU_val = np.mean(metric_res['val'])
        mU_shift1 = np.mean(metric_res['shift1'])
        mU_shift2 = np.mean(metric_res['shift2'])
        mU_shift3 = np.mean(metric_res['shift3'])
        return metric_res, mU_shift1, mU_shift2, mU_shift3

    def train_test_split(self, data_path, dataset, cut_idx, token2index, tk2num, ratio=0.5):
        train_size = int(ratio*(len(dataset)))
        train_idx = random.sample(
            list(np.arange(len(dataset))), train_size
        ) 
        test_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)
        train_idx = np.setdiff1d(train_idx, cut_idx)
        trainset = CodeLoader(data_path, None, token2index, tk2num, train_idx)
        testset = CodeLoader(data_path, None, token2index, tk2num, test_idx)
        return trainset, testset


    def load_data(self, shift1_thre, shift2_thre, shift3_thre, metric_res):
        # retrain idx
        test1_idx = np.where(metric_res['shift1'] > shift1_thre)[0]
        test2_idx = np.where(metric_res['shift2'] > shift2_thre)[0]
        test3_idx = np.where(metric_res['shift3'] > shift3_thre)[0]

        token2index, path2index, func2index, embed, tk2num =\
            perpare_train(
                self.tk_path, self.embed_type, self.vec_path, 
                self.embed_dim, self.res_dir
            )
        index2func = dict2list(func2index)

        # build test loader
        val_dataset = CodeLoader(self.val_data, None, token2index, tk2num)

        test_dataset1 = CodeLoader(self.shift1_data, None, token2index, tk2num)
        shift1_trainset, shift1_testset = self.train_test_split(
            self.shift1_data, test_dataset1, test1_idx, token2index, tk2num
        )
    
        test_dataset2 = CodeLoader(self.shift2_data, None, token2index, tk2num)
        shift2_trainset, shift2_testset = self.train_test_split(
            self.shift2_data, test_dataset2, test2_idx, token2index, tk2num
        )

        test_dataset3 = CodeLoader(self.shift3_data, None, token2index, tk2num)
        shift3_trainset, shift3_testset = self.train_test_split(
            self.shift3_data, test_dataset3, test3_idx, token2index, tk2num
        )

        val_loader = DataLoader(val_dataset, batch_size=self.train_batch, 
                                collate_fn=my_collate)
        test_trainloader1 = DataLoader(shift1_trainset, batch_size=self.train_batch, 
                                       collate_fn=my_collate)
        test_valloader1 = DataLoader(shift1_testset, batch_size=self.train_batch, 
                                     collate_fn=my_collate)
        test_trainloader2 = DataLoader(shift2_trainset, batch_size=self.train_batch, 
                                       collate_fn=my_collate)
        test_valloader2 = DataLoader(shift2_testset, batch_size=self.train_batch, 
                                     collate_fn=my_collate)
        test_trainloader3 = DataLoader(shift3_trainset, batch_size=self.train_batch, 
                                       collate_fn=my_collate)
        test_valloader3 = DataLoader(shift3_testset, batch_size=self.train_batch, 
                                     collate_fn=my_collate)

        return val_loader, test_trainloader1, test_valloader1, test_trainloader2, \
            test_valloader2, test_trainloader3, test_valloader3, index2func


    def retrain(self, retrain_loader_name='shift1'):
        best_val_acc = 0
        total_st_time = datetime.datetime.now()
        print('begin finetuning on {} with metric {}'.format(
            retrain_loader_name, self.metricname
        ))

        for epoch in range(self.start_epoch, self.epochs+1):
            if retrain_loader_name == 'shift1':
                train_model(
                    self.model, epoch, self.test_trainloader1, self.device, 
                    self.criterian, self.optimizer, self.index2func
                )
                res = test_model(
                    self.test_valloader1, self.model, self.device, 
                    self.index2func, 'shift1'
                )
                # save model checkpoint
                if res['shift1 acc'] > best_val_acc:
                    Checkpoint(self.model, self.optimizer, epoch, res).save(self.save_dir)
                    best_val_acc = res['shift1 acc']

            elif retrain_loader_name == 'shift2':
                train_model(
                    self.model, epoch, self.test_trainloader2, self.device, 
                    self.criterian, self.optimizer, self.index2func
                )
                res = test_model(
                    self.test_valloader2, self.model, self.device, 
                    self.index2func, 'shift2'
                )
                # save model checkpoint
                if res['shift2 acc'] > best_val_acc:
                    Checkpoint(self.model, self.optimizer, epoch, res).save(self.save_dir)
                    best_val_acc = res['shift2 acc']
            else:
                train_model(
                    self.model, epoch, self.test_trainloader3, self.device, 
                    self.criterian, self.optimizer, self.index2func
                )
                res = test_model(
                    self.test_valloader3, self.model, self.device, 
                    self.index2func, 'shift3'
                )
                # save model checkpoint
                if res['shift3 acc'] > best_val_acc:
                    Checkpoint(self.model, self.optimizer, epoch, res).save(self.save_dir)
                    best_val_acc = res['shift3 acc']
        
        total_ed_time = datetime.datetime.now()
        print('training {} with metric {} finished! Total cost time: {}'.format(
            retrain_loader_name, self.metricname, total_ed_time - total_st_time
        ))




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Finetune code summary model'
    )
    parser.add_argument('--metric_id', type=int, default=0, choices=[0,1,2,3,4,5],
                        help='index of uncertainty metric')
    parser.add_argument('--res_dir', type=str, default='program_tasks/code_summary/result/java', 
                        help='path to the pretrained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/java_pkl', 
                        help='path to the training/test data')
    parser.add_argument('--metric_dir', type=str, 
                        default='Uncertainty_Results/java/CodeSummary_Module', 
                        help='path to the saved uncertainty result')
    parser.add_argument('--save_dir', type=str, default='Uncertainty_Eval/retrain_res', 
                        help='path to saved the retrained checkpoint and result')
    parser.add_argument('--train_batch', type=int, default=512, help='training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='retraining epochs')
    parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimension of tokens')
    parser.add_argument('--module_id', type=int, default=0, help='index of programming task')
    parser.add_argument('--retrain_data', type=str, default='shift1', 
                        choices=['shift1', 'shift2', 'shift3'],
                        help='retrain data set name')
    
    args = parser.parse_args()
    print(vars(args))

    module_id = args.module_id
    metric_id = args.metric_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_dir = args.res_dir
    data_dir = args.data_dir
    metric_dir = args.metric_dir
    save_dir = args.save_dir
    train_batch = args.train_batch
    epochs = args.epochs
    embed_dim = args.embed_dim
    retrain_data = args.retrain_data

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    retrain = BasicRetrainModule(
        module_id, metric_id, device, res_dir, data_dir, 
        metric_dir, save_dir, train_batch, epochs, embed_dim
    )
    retrain.retrain(retrain_loader_name=retrain_data)

            