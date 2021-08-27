import pickle
from random import sample
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import datetime
import argparse
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, sampler

from preprocess.checkpoint import Checkpoint
from preprocess.utils import set_random_seed
from program_tasks.code_summary.CodeLoader import CodeLoader
from program_tasks.code_summary.Code2VecModule import Code2Vec, CodeBertForClassification2, BiLSTM2Vec
from transformers import RobertaConfig


def my_collate(batch):
    x, y = zip(*batch)
    sts, paths, eds = [], [], []
    for data in x:
        st, path, ed = zip(*data)
        sts.append(torch.tensor(st, dtype=torch.int))
        paths.append(torch.tensor(path, dtype=torch.int))
        eds.append(torch.tensor(ed, dtype=torch.int))

    length = [len(i) for i in sts]
    sts = rnn_utils.pad_sequence(sts, batch_first=True, padding_value=1).long()
    eds = rnn_utils.pad_sequence(eds, batch_first=True, padding_value=1).long()
    paths = rnn_utils.pad_sequence(paths, batch_first=True, padding_value=1).long()
    return (sts, paths, eds), y, length


def dict2list(tk2index):
    res = {}
    for tk in tk2index:
        res[tk2index[tk]] = tk
    return res


def new_acc(pred, y, index2func):
    pred = pred.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    tp, fp, fn = 0, 0, 0
    acc = np.sum(pred == y)
    for i, pred_i in enumerate(pred):
        pred_i = set(index2func[pred_i].split('|'))
        y_i = set(index2func[y[i]].split('|'))
        tp += len(pred_i & y_i)
        fp += len(pred_i - y_i)
        fn = len(y_i - pred_i)
    return acc, tp, fp, fn


def perpare_train(tk_path, embed_type, vec_path, embed_dim, out_dir):
    tk2num = None
    with open(tk_path, 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
        embed = None
    if embed_type == 0: # pretrained embedding
        tk2num, embed = torch.load(vec_path)
        print('load existing embedding vectors, name is ', vec_path)
    elif embed_type == 1: # train with embedding updated
        tk2num = token2index
        print('create new embedding vectors, training from scratch')
    elif embed_type == 2: # train random embedding
        tk2num = token2index
        embed = torch.randn([len(token2index), embed_dim])
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float)
        assert embed.size()[1] == embed_dim
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return token2index, path2index, func2index, embed, tk2num


def train_model(model, train_loader, device,
                criterian, optimizer):
    model.train()

    for i, ((sts, paths, eds), y, length) in tqdm(enumerate(train_loader)):
        sts = sts.to(device)
        paths = paths.to(device)
        eds = eds.to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        
        pred_y = model(starts=sts, paths=paths, ends=eds, length=length)
        loss = criterian(pred_y, y)
        loss.backward()
        optimizer.step()


def test_model(val_loader, model, device, index2func, val_name):
    model.eval()
    acc, tp, fn, fp = 0, 0, 0, 0
    total_samples = 0

    for i, ((sts, paths, eds), y, length) in enumerate(val_loader):
        sts = sts.to(device)
        paths = paths.to(device)
        eds = eds.to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        total_samples += sts.size(0)
        pred_y = model(starts=sts, paths=paths, ends=eds, length=length)
        pos, pred_y = torch.max(pred_y, 1)
        acc_add, tp_add, fp_add, fn_add = new_acc(pred_y, y, index2func)
        tp += tp_add
        fp += fp_add
        fn += fn_add
        acc += acc_add
    
    if total_samples == 0:
        return {f'{val_name} acc': 0}
    acc = acc / total_samples
    prec = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = prec * recall * 2 / (prec + recall + 1e-8) 
    res = {f'{val_name} acc': acc}

    return res


def main(args):
    # parameters setting
    tk_path = args.tk_path
    train_path = args.train_data
    val_path = args.val_data
    # test_path = args.test_data
    test_path1 = args.test_data1
    test_path2 = args.test_data2
    test_path3 = args.test_data3
    # ood_path = args.ood_data
    embed_dim = args.embed_dim
    embed_type = args.embed_type
    vec_path = args.embed_path
    out_dir = args.res_dir
    experiment_name = args.experiment_name
    train_batch = args.batch
    epochs = args.epochs
    lr = args.lr
    weight_decay=args.weight_decay
    max_size = args.max_size
    load_ckpt = args.load_ckpt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data and preparation
    token2index, path2index, func2index, embed, tk2num =\
        perpare_train(tk_path, embed_type, vec_path, embed_dim, out_dir)
    nodes_dim, paths_dim, output_dim = len(tk2num), len(path2index), len(func2index)
    index2func = dict2list(func2index)
    if args.model_type == 'code2vec':
        model = Code2Vec(nodes_dim, paths_dim, embed_dim, output_dim, embed) # modified!
    elif args.model_type == 'codebert':
        pretrained_model = 'microsoft/codebert-base'
        config_class = RobertaConfig
        config_node = config_class.from_pretrained(
            pretrained_model,
            vocab_size=nodes_dim, 
            num_labels=output_dim, 
            use_cache=False, 
            hidden_size=embed_dim
        )
        config_path = config_class.from_pretrained(
            pretrained_model,
            vocab_size=paths_dim, 
            num_labels=output_dim, 
            use_cache=False, 
            hidden_size=embed_dim
        )
        config_concat = config_class.from_pretrained(
            pretrained_model,
            num_labels=output_dim, 
            use_cache=False, 
            hidden_size=3*embed_dim
        )
        model = CodeBertForClassification2([config_node, config_path, config_concat])
    elif args.model_type == 'lstm':
        model = BiLSTM2Vec(nodes_dim, paths_dim, embed_dim, output_dim, embed) # modified!

    criterian = nn.CrossEntropyLoss()  # loss

    # load ckpt if necessary
    if load_ckpt:
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(out_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        start_epoch = resume_checkpoint.epoch
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        start_epoch = 1

    # build test loader
    train_dataset = CodeLoader(train_path, max_size, token2index, tk2num)
    val_dataset = CodeLoader(val_path, max_size, token2index, tk2num)
    # test_dataset = CodeLoader(test_path, max_size, token2index, tk2num)
    test_dataset1 = CodeLoader(test_path1, max_size, token2index, tk2num)
    test_dataset2 = CodeLoader(test_path2, max_size, token2index, tk2num)
    test_dataset3 = CodeLoader(test_path3, max_size, token2index, tk2num)
    # ood_dataset = CodeLoader(ood_path, max_size, token2index, tk2num)

    # train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=my_collate)
    # print('train data size {}, val data size {}, test data size {}'.format(
    #     len(train_dataset), len(val_dataset), len(test_dataset),
    # ))
    print('train data {}, val data {}, test data1 {}, test data2 {}, test data3 {}'.format(
        len(train_dataset), len(val_dataset), len(test_dataset1), 
        len(test_dataset2), len(test_dataset3),
    ))

    train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=train_batch, collate_fn=my_collate)
    # test_loader = DataLoader(test_dataset, batch_size=train_batch, collate_fn=my_collate)
    test_loader1 = DataLoader(test_dataset1, batch_size=train_batch, collate_fn=my_collate)
    test_loader2 = DataLoader(test_dataset2, batch_size=train_batch, collate_fn=my_collate)
    test_loader3 = DataLoader(test_dataset3, batch_size=train_batch, collate_fn=my_collate)
    # ood_loader = DataLoader(ood_dataset, batch_size=train_batch, collate_fn=my_collate)

    # training
    print('begin training experiment {} ...'.format(experiment_name))
    model.to(device)
    best_val_acc = 0
    total_st_time = datetime.datetime.now()

    for epoch in range(start_epoch, epochs+1):
        # print('max size: {}'.format(max_size))
        train_model(model, train_loader, device, criterian, optimizer)
        val_res = test_model(val_loader, model, device, index2func, 'val')
        test_res1 = test_model(test_loader1, model, device, index2func, 'test1')
        test_res2 = test_model(test_loader2, model, device, index2func, 'test2')
        test_res3 = test_model(test_loader3, model, device, index2func, 'test3')
        # ood_res = test_model(ood_loader, model, device, index2func, 'ood')
        merge_res = {**val_res, **test_res1, **test_res2, **test_res3} # merge all the test results
        merge_res["epoch"] = epoch
        print(merge_res)

        # save model checkpoint
        if val_res['val acc'] > best_val_acc:
            Checkpoint(model, optimizer, epoch, merge_res).save(out_dir)
            best_val_acc = val_res['val acc']

    total_ed_time = datetime.datetime.now()
    print('training experiment {} finished! Total cost time: {}'.format(
        experiment_name, total_ed_time - total_st_time
    ))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--classes', default=250, type=int, metavar='N', help='number of output classes')
    parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--model_type', type=str, default='codebert', choices=['codebert', 'code2vec', 'lstm'], help='model architecture')

    parser.add_argument('--embed_dim', default=120, type=int, metavar='N', help='embedding size')
    parser.add_argument('--embed_path', type=str, default='vec/100_2/Doc2VecEmbedding0.vec')
    parser.add_argument('--train_data', type=str, default='data/java_pkl_files/train.pkl')
    parser.add_argument('--val_data', type=str, default='data/java_pkl_files/val.pkl')
    # parser.add_argument('--test_data', type=str, default='data/java_pkl_files/test.pkl')
    parser.add_argument('--test_data1', type=str, default='data/java_pkl_files/test1.pkl')
    parser.add_argument('--test_data2', type=str, default='data/java_pkl_files/test2.pkl')
    parser.add_argument('--test_data3', type=str, default='data/java_pkl_files/test3.pkl')
    # parser.add_argument('--ood_data', type=str, default='python_pkl/test.pkl')
    parser.add_argument('--tk_path', type=str, default='data/java_pkl_files/tk.pkl')
    parser.add_argument('--embed_type', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code summary')
    parser.add_argument('--load_ckpt', default=False, action='store_true', help='load checkpoint')
    parser.add_argument('--res_dir', type=str, default='program_tasks/code_summary/result')
    parser.add_argument('--max_size', type=int, default=None, help='if not None, then use maxsize of the training data')

    args = parser.parse_args()
    options = vars(args)
    print(options)
    # set_random_seed(10)
    main(args)