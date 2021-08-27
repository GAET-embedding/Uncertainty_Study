import pickle
from preprocess.utils import BASEDICT
from tqdm import tqdm
import numpy as np

def build_vocab(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    word2index = BASEDICT.copy()
    for data in dataset:
        print('data: ', data, 'length: ', len(data))
        node_1, _, node_2, _, label = data
        node_combine = node_1 + node_2
        for n in node_combine:
            n = n.lower()
            if n not in word2index:
                word2index[n] = len(word2index)
    return word2index


def tokensize_dataset(file_name, word2index, max_num):
    def tokensize_node(node_list):
        for i, node in enumerate(node_list):
            node_list[i] = word2index[node] if node in word2index else word2index['____UNKNOW____']
        return node_list

    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
        if max_num is not None:
            idx = np.random.randint(0, max_num, max_num)
            dataset = dataset[idx]
    for i, data in tqdm(enumerate(dataset)):
        print('data: ', data)
        node_1, g_1, node_2, g_2, label = data
        node_1 = tokensize_node(node_1)
        node_2 = tokensize_node(node_2)
        dataset[i] = [node_1, g_1, node_2, g_2, label]
    return dataset