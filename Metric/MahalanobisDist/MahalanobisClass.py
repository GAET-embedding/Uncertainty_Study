import torch
from sklearn.linear_model import LogisticRegression

from BasicalClass import BasicModule
from BasicalClass import common_ten2numpy
from Metric import BasicUncertainty
from BasicalClass import IS_DEBUG, DEBUG_NUM


class Mahalanobis(BasicUncertainty):
    def __init__(self, instance: BasicModule, device):
        super(Mahalanobis, self).__init__(instance, device)
        self.hidden_num = 1
        self.u_list, self.std_value = self.preprocess(instance.train_loader)
        self.lr = self.train_logic(instance.train_loader, instance.train_truth)

    def train_logic(self, data_loader, ground_truth):
        train_res = self.extract_metric(data_loader)
        train_res = train_res.reshape([-1, self.hidden_num])
        lr = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
        lr.fit(train_res, ground_truth)
        print(lr.score(train_res, ground_truth))
        return lr

    def preprocess(self, data_loader):
        fx, y = self.get_penultimate(data_loader)
        u_list, std_list = [], []
        for target in range(self.class_num):
            fx_tar = fx[torch.where(y == target)]
            mean_val = torch.mean(fx_tar.float(), dim = 0)
            std_val = (fx_tar - mean_val).transpose(dim0=0, dim1= 1).mm((fx_tar - mean_val))
            u_list.append(mean_val)
            std_list.append(std_val)
        std_value = sum(std_list) / len(y)
        std_value = torch.inverse(std_value)
        return u_list, std_value

    def get_penultimate(self, data_loader):
        # res, y_list = [], []
        # for i, (x, y) in enumerate(data_loader):
        #     x = x.to(self.device)
        #     self.model.to(self.device)
        #     fx = self.model.get_feature(x)
        #     res.append(fx)
        #     y_list.append(y)
        #     if IS_DEBUG and i >= DEBUG_NUM:
        #         break
        # res = torch.cat(res, dim=0)
        # y_list = torch.cat(y_list, dim=0)
        # return res, y_list
        pred_pos, pred_list, y_list = [], [], []
        self.model.to(self.device)

        for i, ((sts, paths, eds), y, length) in enumerate(data_loader):
            torch.cuda.empty_cache()
            sts = sts.to(self.device)
            paths = paths.to(self.device)
            eds = eds.to(self.device)
            y = torch.tensor(y, dtype=torch.long)
            output = self.model(sts, paths, eds, length, self.device)
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
            return torch.cat(y_list, dim=0), torch.cat(pred_list, dim=0)

    def extract_metric(self, data_loader):
        fx, _ = self.get_penultimate(data_loader)
        score = []
        for target in range(self.class_num):
            tmp = (fx - self.u_list[target]).mm(self.std_value)
            tmp = tmp.mm((fx - self.u_list[target]).transpose(dim0=0, dim1=1) )
            tmp = tmp.diagonal().reshape([-1, 1])
            score.append(-tmp)
        score = torch.cat(score, dim=1)
        score = common_ten2numpy(torch.max(score, dim=1)[0])
        return score

    def _uncertainty_calculate(self, data_loader):
        metric = self.extract_metric(data_loader).reshape([-1, self.hidden_num])
        result = self.lr.predict_proba(metric)[:, 1]
        return result

