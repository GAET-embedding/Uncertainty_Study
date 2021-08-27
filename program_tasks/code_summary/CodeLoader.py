  
from torch.utils.data import Dataset
import pickle
import random
import numpy as np

class CodeLoader(Dataset):
    def dict2list(self, token2index):
        # index2token (0 -> UNK, 1 -> PAD)
        res = {}
        for k in token2index:
            res[token2index[k]] = k 
        return res 

    def toktransfer(self, context):
        if context not in self.index2token: # OOD token
            return self.tk2num['____UNKNOW____']
        else:
            tk = ''.join(self.index2token[context].split('|')).lower()
            return self.tk2num[tk] if tk in self.tk2num else self.tk2num['____UNKNOW____']

    def pathtransfer(self, path):
        if self.path2index is None:
            return path
        else:
            if path not in self.index2path: # OOD path
                return self.path2index['____UNKNOW____']
            else:
                return path

    def functransfer(self, label):
        if self.func2index is None:
            return label
        else:
            if label not in self.index2func: # OOD label
                return self.func2index['____UNKNOW____']
            else:
                return label
    

    def transferdata(self):
        for i, data in enumerate(self.dataset):
            code_context, target = data
            code_context = [
                [
                    self.toktransfer(context[0]), 
                    self.pathtransfer(context[1]), 
                    self.toktransfer(context[2])
                ]
                for context in code_context
            ]
            self.dataset[i] = [code_context, self.functransfer(target)]

    def __init__(
        self, 
        file_name, 
        max_size, 
        token2index, 
        tk2num, 
        idx=None, 
        path2index=None, 
        func2index=None,
    ):
        self.path2index = path2index
        self.index2path = None
        if path2index is not None:
            self.index2path = self.dict2list(path2index)

        self.func2index = func2index
        self.index2func = None
        if func2index is not None:
            self.index2func = self.dict2list(func2index)
        
        self.tk2num = tk2num
        self.index2token = self.dict2list(token2index)
        with open(file_name, 'rb') as f:
            self.dataset = pickle.load(f)
            # random.shuffle(self.dataset)
            
        if max_size is not None:
            self.dataset = self.dataset[:max_size]
        if tk2num is not None:
            self.transferdata()
        if idx is not None:
            self.dataset = list(np.array(self.dataset)[idx])
            

    def __getitem__(self, index):
        data = self.dataset[index]
        code_context, target = data
        return code_context, target

    def __len__(self):
        return len(self.dataset)