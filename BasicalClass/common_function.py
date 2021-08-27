import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc, average_precision_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cpu':
    print('Using CPU ...')
    # os.chdir('F:/Python/UncertaintyStudy')  # This set the current work directory.
    os.chdir('.')  # This set the current work directory.
    IS_DEBUG = True
else:
    print('Using CUDA GPU ...')
    IS_DEBUG = False
DEBUG_NUM = 10
RAND_SEED = 333


def common_predict_y(dataset, model, device, batch_size = 32, ): # Todo : modeify the batch_size to a large number
    model.to(device)
    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=None,
    )
    pred_pos, pred_y, y_list = [], [], []
    for i, (x, y_label) in enumerate(data_loader):
        output = model(x.to(device))
        pos, y = torch.max(output, dim=1)
        pred_pos.append(output.detach())
        pred_y.append(y.detach())
        y_list.append(y_label)
        if IS_DEBUG and i >= DEBUG_NUM:
            break
    return  torch.cat(pred_pos, dim = 0).to(device), \
            torch.cat(pred_y, dim=0).view([-1]).to(device),\
            torch.cat(y_list, dim = 0).view([-1]).to(device)


def common_predict(data_loader, model, device, train_sub=False, module_id=0):
    pred_pos, pred_list, y_list = [], [], []

    with torch.no_grad():
        if train_sub: # train sub linear fc model
            for i, (x, y) in enumerate(data_loader):
                torch.cuda.empty_cache()
                x = x.to(device)
                output = model(x)
                _, pred_y = torch.max(output, dim=1) # pos, pred_y
                y = torch.tensor(y, dtype=torch.long)
                # detach
                x = x.detach().cpu()
                pred_y = pred_y.detach().cpu()
                output = output.detach().cpu()

                pred_list.append(pred_y)
                pred_pos.append(output)
                y_list.append(y)

                if IS_DEBUG and i >= DEBUG_NUM:
                    break
            
        else:
            model.to(device)
            model.eval()

            if module_id == 0: # code summary
                # print('calculating code summary ...')
                for i, ((sts, paths, eds), y, length) in enumerate(data_loader):
                    torch.cuda.empty_cache()
                    sts = sts.to(device)
                    paths = paths.to(device)
                    eds = eds.to(device)
                    y = torch.tensor(y, dtype=torch.long)
                    output = model(starts=sts, paths=paths, ends=eds, length=length)

                    _, pred_y = torch.max(output, dim=1)
                    # detach
                    sts = sts.detach().cpu()
                    paths = paths.detach().cpu()
                    eds = eds.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    output = output.detach().cpu()

                    pred_list.append(pred_y)
                    pred_pos.append(output)
                    y_list.append(y)

                    if IS_DEBUG and i >= DEBUG_NUM:
                        break
            
            elif module_id == 1: # code completion
                # print('calculating code completion ...')
                for i, (input, y, _) in tqdm(enumerate(data_loader)):
                    torch.cuda.empty_cache()
                    input = input.to(device)
                    # compute output
                    output = model(input)
                    _, pred_y = torch.max(output, dim=1)
                    # detach
                    input = input.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    output = output.detach().cpu()
                
                    # measure accuracy and record loss
                    pred_list.append(pred_y)
                    pred_pos.append(output)
                    y_list.append(y.long())

                    if IS_DEBUG and i >= DEBUG_NUM:
                        break
            else:
                raise TypeError()

    return torch.cat(pred_pos, dim=0), torch.cat(pred_list, dim = 0), torch.cat(y_list, dim = 0)

    
def common_get_auc(y_test, y_score, name=None):
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    if name is not None:
        print(name, 'auc is ', roc_auc)
    return roc_auc


def common_plotROC(y_test, y_score, file_name= None):
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # if file_name is not None:
    #     plt.savefig(file_name)
    # else:
    #     plt.show()
    print(file_name, 'auc is ', roc_auc)
    return roc_auc

def common_get_aupr(y_test, y_score, name=None):
    aupr = average_precision_score(y_test, y_score)
    if name is not None:
        print(name, 'aupr is ', aupr)
    return aupr


def common_get_accuracy(ground_truth, oracle_pred, threshhold = 0.1):
    oracle_pred = (oracle_pred > threshhold)
    pos_acc = (np.sum((oracle_pred == 1) * (ground_truth == 1))) / (np.sum(oracle_pred == 1) + 1)
    neg_acc = (np.sum((oracle_pred == 0) * (ground_truth == 1))) / (np.sum(oracle_pred == 0) + 1)
    coverage = (np.sum((oracle_pred == 1) * (ground_truth == 1))) / (np.sum(ground_truth == 1) + 1)
    print(threshhold, pos_acc, neg_acc, coverage)


def common_get_xy(dataset, batch_size, device):
    x,y = [],[]
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    for i, data in enumerate(data_loader):
        x.append(data[0])
        y.append(data[1])
        if IS_DEBUG and i >= DEBUG_NUM:
            break
    return torch.cat(x, dim = 0).cpu(), torch.cat(y,dim = 0).cpu()


def common_cal_accuracy(pred_y, y):
    tmp = (pred_y.view([-1]) == y.view([-1]))
    acc = torch.sum(tmp.float()) / len(y)
    return acc


def common_load_corroptions():
    dir_name = './data/cifar_10/CIFAR-10-C/'
    y = np.load(dir_name + 'labels.npy')
    y = torch.tensor(y, dtype=torch.long)
    for file_name in os.listdir(dir_name):
        if file_name != 'labels.npy':
            x = np.load(dir_name + file_name)
            yield x, y, file_name.split('.')[0]


def common_get_maxpos(pos : torch.Tensor):
    test_pred_pos, _ = torch.max(F.softmax(pos, dim=1), dim=1)
    return common_ten2numpy(test_pred_pos)

def common_get_entropy(pos : torch.Tensor):
    k = pos.size(-1)
    pred_prob = F.softmax(pos, dim=-1) # (N, k)
    etp = entropy(pred_prob, axis=-1)/np.log(k) # np.ndarray
    return 1 - etp


def common_ten2numpy(a:torch.Tensor):
    return a.detach().cpu().numpy()