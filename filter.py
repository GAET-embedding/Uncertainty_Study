import os
import numpy as np
from torch.cuda import is_available
import torch
import torch.nn as nn
from BasicalClass.common_function import *
from program_tasks.code_summary.CodeLoader import CodeLoader
from program_tasks.code_summary.main import perpare_train, my_collate, test_model, dict2list
from program_tasks.code_completion.vocab import VocabBuilder
from program_tasks.code_completion.dataloader import Word2vecLoader
from program_tasks.code_completion.main import test
from preprocess.checkpoint import Checkpoint
from tqdm import tqdm


class Filter:
    def __init__(self, res_dir, data_dir, metric_dir, save_dir, 
                 device, module_id, shift, max_size, batch_size):

        self.res_dir = res_dir
        self.data_dir = data_dir
        self.device = device
        self.shift = shift
        self.module_id = module_id
        self.embed_type = 1
        self.vec_path = None
        self.embed_dim = 120
        self.max_size = max_size
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.vanilla = torch.load(os.path.join(metric_dir, 'Vanilla.res'))
        self.temp = torch.load(os.path.join(metric_dir, 'ModelWithTemperature.res'))
        self.pv = torch.load(os.path.join(metric_dir, 'PVScore.res'))
        self.dropout = torch.load(os.path.join(metric_dir, 'ModelActivateDropout.res'))
        self.mutation = torch.load(os.path.join(metric_dir, 'Mutation.res'))
        self.batch_size = batch_size

        if module_id == 0: # code summary
            self.tk_path = os.path.join(self.data_dir, 'tk.pkl')
            self.train_path = os.path.join(self.data_dir, 'train.pkl')
            self.val_path = os.path.join(self.data_dir, 'val.pkl')
            if shift:
                self.test_path = None
                self.test1_path = os.path.join(self.data_dir, 'test1.pkl')
                self.test2_path = os.path.join(self.data_dir, 'test2.pkl')
                self.test3_path = os.path.join(self.data_dir, 'test3.pkl')
            else:
                self.test_path = os.path.join(self.data_dir, 'test.pkl')
                self.test1_path = None
                self.test2_path = None
                self.test3_path = None
        else: # code completion
            self.tk_path = None
            self.train_path = os.path.join(self.data_dir, 'train.tsv')
            self.val_path = os.path.join(self.data_dir, 'val.tsv')
            if shift:
                self.test_path = None
                self.test1_path = os.path.join(self.data_dir, 'test1.tsv')
                self.test2_path = os.path.join(self.data_dir, 'test2.tsv')
                self.test3_path = os.path.join(self.data_dir, 'test3.tsv')
            else:
                self.test_path = os.path.join(self.data_dir, 'test.tsv')
                self.test1_path = None
                self.test2_path = None
                self.test3_path = None

        # load ckpt 
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        self.model = resume_checkpoint.model
        self.model.to(self.device)
        self.model.eval()

        if module_id == 0: # code summary
            # load data and preparation
            self.token2index, path2index, func2index, embed, self.tk2num = perpare_train(
                self.tk_path, self.embed_type, self.vec_path, 
                self.embed_dim, self.res_dir
            )
            self.index2func = dict2list(func2index)
            
            # build test loader
            if shift:
                self.test_dataset1 = CodeLoader(
                    self.test1_path, self.max_size, self.token2index, self.tk2num
                )
                self.test_dataset2 = CodeLoader(
                    self.test2_path, self.max_size, self.token2index, self.tk2num
                )
                self.test_dataset3 = CodeLoader(
                    self.test3_path, self.max_size, self.token2index, self.tk2num
                )
            else:
                self.test_dataset = CodeLoader(
                    self.test_path, self.max_size, self.token2index, self.tk2num
                )
        else: # code completion
            min_samples = 5
            # load data and preparation
            v_builder = VocabBuilder(path_file=self.train_path)
            self.d_word_index, embed = v_builder.get_word_index(min_sample=min_samples)

            if embed is not None:
                if type(embed) is np.ndarray:
                    embed = torch.tensor(embed, dtype=torch.float).cuda()
                assert embed.size()[1] == self.embed_dim


    def filtering(self, res, coverage, testset='val'):
        original_size = len(self.vanilla[testset])
        remain_size = int(coverage * original_size)
        if testset == 'val':
            data_path = self.val_path
        elif testset == 'test1':
            data_path = self.test1_path
        elif testset == 'test2':
            data_path = self.test2_path
        elif testset == 'test3':
            data_path = self.test3_path
        elif testset == 'test':
            data_path = self.test_path
        else:
            raise ValueError()

        # vanilla
        va_idx = np.argsort(self.vanilla[testset])[::-1][:remain_size]
        # print("vanilla all index ({}), selected index ({}), max index {}".format(
        #     len(self.vanilla[testset]), len(va_idx), max(va_idx)
        # ))
        if self.module_id == 0: # code summary
            va_dataset = CodeLoader(data_path, self.max_size, self.token2index, self.tk2num, idx=va_idx)
            va_test_loader = DataLoader(va_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            va_acc = test_model(va_test_loader, self.model, self.device, self.index2func, testset)[f'{testset} acc']
        else: # code completion
            va_test_loader = Word2vecLoader(
                data_path, self.d_word_index, batch_size=self.batch_size, 
                max_size=self.max_size, idx=list(va_idx)
            )
            va_acc = test(va_test_loader, self.model, testset)[f'{testset} acc']
    
        res['va_acc'][testset].append(va_acc)

        # temp scaling
        temp_idx = np.argsort(self.temp[testset])[::-1][:remain_size]
        if self.module_id == 0: # code summary
            temp_dataset = CodeLoader(data_path, self.max_size, self.token2index, self.tk2num, idx=temp_idx)
            temp_test_loader = DataLoader(temp_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            temp_acc = test_model(temp_test_loader, self.model, self.device, self.index2func, testset)[f'{testset} acc']
        else: # code completion
            temp_test_loader = Word2vecLoader(
                data_path, self.d_word_index, batch_size=self.batch_size, 
                max_size=self.max_size, idx=list(temp_idx)
            )
            temp_acc = test(temp_test_loader, self.model, testset)[f'{testset} acc']

        res['temp_acc'][testset].append(temp_acc)

        # mutation
        mutation_idx = np.argsort(self.mutation[testset][0])[::-1][:remain_size]
        if self.module_id == 0: # code summary
            mutation_dataset = CodeLoader(data_path, self.max_size, self.token2index, self.tk2num, idx=mutation_idx)
            mutation_test_loader = DataLoader(mutation_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            mutation_acc = test_model(mutation_test_loader, self.model, self.device, self.index2func, testset)[f'{testset} acc']
        else: # code completion
            mutation_test_loader = Word2vecLoader(
                data_path, self.d_word_index, batch_size=self.batch_size, 
                max_size=self.max_size, idx=list(mutation_idx)
            )
            mutation_acc = test(mutation_test_loader, self.model, testset)[f'{testset} acc']

        res['mutation_acc'][testset].append(mutation_acc)

        # dropout
        dropout_idx = np.argsort(self.dropout[testset])[::-1][:remain_size]
        if self.module_id == 0: # code summary
            dropout_dataset = CodeLoader(data_path, self.max_size, self.token2index, self.tk2num, idx=dropout_idx)
            dropout_test_loader = DataLoader(dropout_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            dropout_acc = test_model(dropout_test_loader, self.model, self.device, self.index2func, testset)[f'{testset} acc']
        else: # code completion
            dropout_test_loader = Word2vecLoader(
                data_path, self.d_word_index, batch_size=self.batch_size, 
                max_size=self.max_size, idx=list(dropout_idx)
            )
            dropout_acc = test(dropout_test_loader, self.model, testset)[f'{testset} acc']

        res['dropout_acc'][testset].append(dropout_acc)

        # dissector
        pv_idx = np.argsort(self.pv[testset][0])[::-1][:remain_size]
        if self.module_id == 0: # code summary
            pv_dataset = CodeLoader(data_path, self.max_size, self.token2index, self.tk2num, idx=pv_idx)
            pv_test_loader = DataLoader(pv_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            pv_acc = test_model(pv_test_loader, self.model, self.device, self.index2func, testset)[f'{testset} acc']
        else: # code completion
            pv_test_loader = Word2vecLoader(
                data_path, self.d_word_index, batch_size=self.batch_size, 
                max_size=self.max_size, idx=list(pv_idx)
            )
            pv_acc = test(pv_test_loader, self.model, testset)[f'{testset} acc']
        res['pv_acc'][testset].append(pv_acc)

        print('{} set [coverage {}]: vanilla test acc {}, temp test acc {}, mutation test acc {}, dropout test acc {}, dissector test acc {}'.format(
            testset, coverage, va_acc, temp_acc, mutation_acc, dropout_acc, pv_acc
        ))

    def run(self):
        # evaluate on test dataset
        coverage_range = np.arange(0.01, 1.01, 0.01)
        res = {
            'x': coverage_range, 
            'va_acc': {}, 
            'temp_acc': {}, 
            'mutation_acc': {},
            'dropout_acc': {},
            'pv_acc': {},
        }
    
        testsets = ['val', 'test1', 'test2', 'test3']
        for testset in testsets:
            res['va_acc'][testset] = []
            res['temp_acc'][testset] = []
            res['mutation_acc'][testset] = []
            res['dropout_acc'][testset] = []
            res['pv_acc'][testset] = []
            for coverage in tqdm(coverage_range):
                self.filtering(res, coverage, testset)
            
        # save file 
        torch.save(res, os.path.join(self.save_dir, 'filter.res'))



if __name__ == "__main__":

    shift_type = 'different_time'
    module_id = 0 # code summary
    # module_id = 1 # code completion
    # model_type = 'code2vec'
    # model_type = 'lstm'
    model_type = 'codebert'
    # model_type = 'word2vec'
    uncertainty_dir = 'Uncertainty_Results_new'

    task = 'code_summary' if module_id == 0 else 'code_completion'
    module_dir = 'CodeSummary_Module' if module_id == 0 else 'CodeCompletion_Module'
    if shift_type == 'different_author':
        if module_id == 0:
            res_dir = f'program_tasks/{task}/result/{shift_type}/elasticsearch/java_project/{model_type}'
            data_dir = f'java_data/{shift_type}/elasticsearch/java_pkl'
        else:
            res_dir = f'program_tasks/{task}/result/{shift_type}/elasticsearch/{model_type}'
            data_dir = f'program_tasks/{task}/dataset/{shift_type}/elasticsearch'
        metric_dir = f'{uncertainty_dir}/{shift_type}/elasticsearch/{model_type}/{module_dir}'
    else:
        res_dir = f'program_tasks/{task}/result/{shift_type}/java_project/{model_type}'
        if module_id == 0:
            data_dir = f'java_data/{shift_type}/java_pkl'
        else:
            data_dir = f'program_tasks/{task}/dataset/{shift_type}/java_project'
        metric_dir = f'{uncertainty_dir}/{shift_type}/java_project/{model_type}/{module_dir}'
    
    save_dir = f'Uncertainty_Eval/filter/{shift_type}/{model_type}/{module_dir}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    shift = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_size = None if module_id == 0 else 200
    batch_size = 256 if module_id == 0 else 64

    filter = Filter(
        res_dir=res_dir, data_dir=data_dir, metric_dir=metric_dir,
        save_dir=save_dir, device=device, module_id=module_id, shift=shift,
        max_size=max_size, batch_size=batch_size
    )

    filter.run()






