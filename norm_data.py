import pickle
import json
import os
from preprocess.utils import BASEDICT


def build_dict(dataset):
    base_dict = BASEDICT.copy()
    token2index, path2index, func2index = base_dict.copy(), base_dict.copy(), base_dict.copy()
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        if func_name not in func2index:
            func2index[func_name] = len(func2index)
        for context in code_context:
            st, path, ed = context.split(',')
            if st not in token2index:
                token2index[st] = len(token2index)
            if ed not in token2index:
                token2index[ed] = len(token2index)
            if path not in path2index:
                path2index[path] = len(path2index)
    with open(DIR + '/' + 'tk.pkl', 'wb') as f:
        pickle.dump([token2index, path2index, func2index], f)
    print("finish dictionary build", len(token2index), len(path2index), len(func2index))


def tk2index(tk_dict, k):
    if k not in tk_dict:
        return tk_dict['____UNKNOW____']
    return tk_dict[k]


def norm_data(data_type, suffix='.c2v'):
    file_name = DATA_DIR + '/'+ FILENAME + '.' + data_type + suffix
    with open(file_name, 'r') as f:
        dataset = f.readlines()
    with open(DIR + '/' + 'tk.pkl', 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
    newdataset = []
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        label = tk2index(func2index, func_name)
        newdata = []
        for context in code_context:
            st, path, ed = context.split(',')
            newdata.append(
                [tk2index(token2index, st), tk2index(path2index, path), tk2index(token2index, ed)]
            )
        newdataset.append([newdata, label])
    save_file = DIR + '/' + data_type + '.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(newdataset, f)
    print("finish normalize dataset", data_type)


def main(DATA_DIR, DIR, suffix='.c2v'):
    with open(DATA_DIR + '/' + FILENAME + '.train' + suffix, 'r') as f:
        dataset = f.readlines()
        print('dataset number is ', len(dataset))
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    build_dict(dataset)
    norm_data('train', suffix=suffix)
    norm_data('val', suffix=suffix)
    norm_data('test', suffix=suffix)
    # norm_data('test1', suffix=suffix)
    # norm_data('test2', suffix=suffix)
    # norm_data('test3', suffix=suffix)


if __name__ == '__main__':
    file_dict = {
        # 'elasticsearch': 'java_pkl',
        # 'java_project1': 'java_pkl1',
        # 'java_project2': 'java_pkl2',
        # 'java_project3': 'java_pkl3',
        # 'java_project': 'java_pkl'
        'python_files': 'python_pkl'
    }
    for FILENAME, GEN_FILE in file_dict.items():
        # TRG_DIR = 'java_data/different_author/elasticsearch'
        # TRG_DIR = 'java_data/different_time'
        # TRG_DIR = 'java_data/different_project'
        TRG_DIR = '' 
        DIR = os.path.join(TRG_DIR, GEN_FILE)
        DATA_DIR = os.path.join(TRG_DIR, FILENAME)
        main(DATA_DIR, DIR, suffix='.c2s')
