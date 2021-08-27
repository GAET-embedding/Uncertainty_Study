import os
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, brier_score_loss, precision_recall_curve


class Uncertainty_Eval():
    def __init__(self, res_dir, save_dir, task='CodeSummary_Module', shift=False, ood=False):
        """
        res_dir (str): path of uncertainty result, Default: "Uncertainty_Results".
        save_dir (str): path of saving evaluation res, Default: "Uncertainty_Eval/java".
        task (str): task name like CodeSummary_Module.
        """
        self.res_dir = res_dir
        self.task = task
        self.save_dir = save_dir
        self.shift = shift
        self.ood = ood
        
    def common_get_auc(self, y_test, y_score):
        # calculate true positive & false positive
        try:
            fpr, tpr, threshold = roc_curve(y_test, y_score)  
            roc_auc = auc(fpr, tpr)  # calculate AUC
            return roc_auc 
        except:
            return 0.0

    def common_get_aupr(self, y_test, y_score):
        try:
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)
            area = auc(recall, precision)
            return area
        except:
            return 0.0

    def common_get_nll(self, y_test, y_score):
        pred_logits = torch.cat((
            torch.tensor(y_score).unsqueeze(1), 
            torch.tensor(1-y_score).unsqueeze(1)
        ), dim=-1)
        nll = torch.nn.NLLLoss()
        return nll(pred_logits, torch.tensor(y_test).long()).item()

    def common_get_brier(self, y_test, y_score):
        try:
            brier = brier_score_loss(y_test, y_score)
            return brier
        except:
            return 1.0

    def common_cal(self, y_test, y_score, metric='AUC'):
        if metric.lower() == 'auc':
            return self.common_get_auc(y_test, y_score)
        elif metric.lower() == 'aupr':
            return self.common_get_aupr(y_test, y_score)
        elif metric.lower() == 'brier':
            return self.common_get_brier(y_test, y_score)
        else:
            raise TypeError("Unknown metric type!")

    def cal_mUncertainty(self, metric_name, metric_res, eval_res):
        if metric_name not in ['Mutation', 'PVScore']:
            mU_val = np.mean(metric_res['val'])
            if self.ood:
                mU_ood = np.mean(metric_res['ood'])
            if not self.shift:
                mU_test = np.mean(metric_res['test'])
                if not self.ood:
                    print('%s: \nmUncertainty: val: %.4f, test: %.4f' % (metric_name, mU_val, mU_test))
                    eval_res[metric_name]['mUncertain'] = {'val': mU_val, 'test': mU_test}
                else:
                    print('%s: \nmUncertainty: val: %.4f, test: %.4f, ood: %.4f' % (metric_name, mU_val, mU_test, mU_ood))
                    eval_res[metric_name]['mUncertain'] = {'val': mU_val, 'test': mU_test, 'ood': mU_ood}
            else:
                mU_test1 = np.mean(metric_res['test1'])
                mU_test2 = np.mean(metric_res['test2'])
                mU_test3 = np.mean(metric_res['test3'])
                if not self.ood:
                    print('%s: \nmUncertainty: val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f' % (
                        metric_name, mU_val, mU_test1, mU_test2, mU_test3
                    ))
                    eval_res[metric_name]['mUncertain'] = {
                        'val': mU_val, 'test1': mU_test1,
                        'test2': mU_test2, 'test3': mU_test3,
                    }
                else:
                    print('%s: \nmUncertainty: val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f, ood: %.4f' % (
                        metric_name, mU_val, mU_test1, mU_test2, mU_test3, mU_ood
                    ))
                    eval_res[metric_name]['mUncertain'] = {
                        'val': mU_val, 'test1': mU_test1,
                        'test2': mU_test2, 'test3': mU_test3, 'ood': mU_ood
                    }
        else:
            # average uncertainty
            mU_vals = [np.mean(res) for res in metric_res['val']]
            if self.ood:
                mU_oods = [np.mean(res) for res in metric_res['ood']]
            if not self.shift:
                mU_tests = [np.mean(res) for res in metric_res['test']]
                count = 0
                if not self.ood:
                    for mU_val, mU_test in zip(mU_vals, mU_tests):
                        count += 1
                        print('%s (method %d): \nmUncertainty: val: %.4f, test: %.4f' % (
                            metric_name, count, mU_val, mU_test
                        ))
                    eval_res[metric_name]['mUncertain'] = [
                        {'val': mU_val, 'test': mU_test}
                        for mU_val, mU_test in zip(mU_vals, mU_tests)
                    ]
                else:
                    for mU_val, mU_test, mU_ood in zip(mU_vals, mU_tests, mU_oods):
                        count += 1
                        print('%s (method %d): \nmUncertainty: val: %.4f, test: %.4f, ood: %.4f' % (
                            metric_name, count, mU_val, mU_test, mU_ood
                        ))
                    eval_res[metric_name]['mUncertain'] = [
                        {'val': mU_val, 'test': mU_test, 'ood': mU_ood}
                        for mU_val, mU_test in zip(mU_vals, mU_tests, mU_oods)
                    ]
            else:
                mU1s = [np.mean(res) for res in metric_res['test1']]
                mU2s = [np.mean(res) for res in metric_res['test2']]
                mU3s = [np.mean(res) for res in metric_res['test3']]
                count = 0
                if not self.ood:
                    for mU_val, mU1, mU2, mU3 in zip(mU_vals, mU1s, mU2s, mU3s):
                        count += 1
                        print('%s (method %d): \nmUncertainty: val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f' % (
                            metric_name, count, mU_val, mU1, mU2, mU3
                        ))
                    eval_res[metric_name]['mUncertain'] = [
                        {'val': mU_val, 'test1': mU1, 'test2': mU2, 'test3': mU3}
                        for mU_val, mU1, mU2, mU3 in zip(mU_vals, mU1s, mU2s, mU3s)
                    ]
                else:
                    for mU_val, mU1, mU2, mU3, mU_ood in zip(mU_vals, mU1s, mU2s, mU3s, mU_oods):
                        count += 1
                        print('%s (method %d): \nmUncertainty: val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f, ood: %.4f' % (
                            metric_name, count, mU_val, mU1, mU2, mU3, mU_ood
                        ))
                    eval_res[metric_name]['mUncertain'] = [
                        {'val': mU_val, 'test1': mU1, 'test2': mU2, 'test3': mU3, 'ood': mU_ood}
                        for mU_val, mU1, mU2, mU3, mU_ood in zip(mU_vals, mU1s, mU2s, mU3s, mU_oods)
                    ]


    def cal_metric(self, metric_name, truth, metric_res, eval_res, metric='AUC'):
        if metric_name not in ['Mutation', 'PVScore']:
            metric_val = self.common_cal(truth['val'], metric_res['val'], metric)
            if not self.shift:
                metric_test = self.common_cal(truth['test'], metric_res['test'], metric)
                print('%s: val: %.4f, test: %.4f' % (metric, metric_val, metric_test))
                eval_res[metric_name][metric] = {'val': metric_val, 'test': metric_test}
            else:
                metric_test1 = self.common_cal(truth['test1'], metric_res['test1'], metric)
                metric_test2 = self.common_cal(truth['test2'], metric_res['test2'], metric)
                metric_test3 = self.common_cal(truth['test3'], metric_res['test3'], metric)
                print('%s: val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f' % (
                    metric, metric_val, metric_test1, metric_test2, metric_test3
                ))
                eval_res[metric_name][metric] = {
                    'val': metric_val, 'test1': metric_test1, 
                    'test2': metric_test2, 'test3': metric_test3,
                }
        else: # we pick the best eval result for PVScore and Mutation
            metric_vals = [
                self.common_cal(truth['val'], res, metric) for res in metric_res['val']
            ]
            if not self.shift:
                metric_tests = [
                    self.common_cal(truth['test'], res, metric) for res in metric_res['test']
                ]
                count = 0
                for metric_val, metric_test in zip(metric_vals, metric_tests):
                    count += 1
                    print('%s (method %d): val: %.4f, test: %.4f' % (
                        metric, count, metric_val, metric_test
                    ))
                eval_res[metric_name][metric] = [
                    {'val': metric_val, 'test': metric_test} 
                    for metric_val, metric_test in zip(metric_vals, metric_tests)
                ]
            else:
                mts1 = [
                    self.common_cal(truth['test1'], res, metric) for res in metric_res['test1']
                ]
                mts2 = [
                    self.common_cal(truth['test2'], res, metric) for res in metric_res['test2']
                ]
                mts3 = [
                    self.common_cal(truth['test3'], res, metric) for res in metric_res['test3']
                ]
                count = 0
                for metric_val, mt1, mt2, mt3 in zip(metric_vals, mts1, mts2, mts3):
                    count += 1
                    print('%s (method %d): val: %.4f, test1: %.4f, test2: %.4f, test3: %.4f' % (
                        metric, count, metric_val, mt1, mt2, mt3
                    ))
                eval_res[metric_name][metric] = [
                    {'val': metric_val, 'test1': mt1, 'test2': mt2, 'test3': mt3}
                    for metric_val, mt1, mt2, mt3 in zip(metric_vals, mts1, mts2, mts3)
                ]


    def evaluation(self):
        eval_res = {}
        src_dir = os.path.join(self.res_dir, self.task)
        truth = torch.load(os.path.join(src_dir,'truth.res'))
        uncertainty_res = [f for f in os.listdir(src_dir) if f.endswith('.res') and f != 'truth.res']
        
        if not self.shift:
            if not self.ood:
                print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f' % (
                    np.mean(truth['train']), np.mean(truth['val']), np.mean(truth['test'])
                ))
            else:
                print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f, ood_acc: %.4f' % (
                    np.mean(truth['train']), np.mean(truth['val']), 
                    np.mean(truth['test']), np.mean(truth['ood'])
                ))
        for metric in uncertainty_res:
            metric_res = torch.load(os.path.join(src_dir, metric))
            metric_name = metric[:-4] # get rid of endswith '.res'
            eval_res[metric_name] = {}

            # average uncertainty
            self.cal_mUncertainty(metric_name, metric_res, eval_res)
            # AUC
            self.cal_metric(metric_name, truth, metric_res, eval_res, 'AUC')
            # AUPR
            self.cal_metric(metric_name, truth, metric_res, eval_res, 'AUPR')
            # Brier score
            self.cal_metric(metric_name, truth, metric_res, eval_res, 'Brier')

        # save evaluation res
        if not os.path.exists(os.path.join(self.save_dir, self.task)):
            os.makedirs(os.path.join(self.save_dir, self.task))
        save_name = os.path.join(self.save_dir, self.task, 'uncertainty_eval.res')
        torch.save(eval_res, save_name)

    
    def ood_detect(self):
        ood_res = {}
        src_dir = os.path.join(self.res_dir, self.task)
        truth = torch.load(os.path.join(src_dir,'truth.res'))
        uncertainty_res = [f for f in os.listdir(src_dir) if f.endswith('.res') and f != 'truth.res']
        
        if not self.shift:
            print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f, ood_acc: %.4f' % (
                np.mean(truth['train']), np.mean(truth['val']), 
                np.mean(truth['test']), np.mean(truth['ood'])
            ))
        else:
            print('train_acc: %.4f, val_acc: %.4f, test1_acc: %.4f, test2_acc: %.4f, test3_acc: %.4f, ood_acc: %.4f' % (
                np.mean(truth['train']), np.mean(truth['val']), 
                np.mean(truth['test1']), np.mean(truth['test2']), 
                np.mean(truth['test3']), np.mean(truth['ood'])
            ))

        # val as in-distribution, ood as out-of-distribution
        oracle = np.array([1]*len(truth['val']) + [0]*len(truth['ood']))
        # oracle = np.array([1]*len(truth['test1']) + [0]*len(truth['ood']))
        # print("in_data {} ood_data {}".format(len(truth['val']), len(truth['ood'])))

        for metric in uncertainty_res:
            metric_res = torch.load(os.path.join(src_dir, metric))
            metric_name = metric[:-4] # get rid of endswith '.res'
            ood_res[metric_name] = {}

            # average uncertainty
            self.cal_mUncertainty(metric_name, metric_res, ood_res)

            if metric_name not in ['Mutation', 'PVScore']:
                pred = np.concatenate((metric_res['val'], metric_res['ood']))
                # pred = np.concatenate((metric_res['test1'], metric_res['ood']))
                AUC = self.common_get_auc(oracle, pred) # AUC
                AUPR = self.common_get_aupr(oracle, pred) # AUPR
                Brier = self.common_get_brier(oracle, pred) # Brier score
                print('AUC: %.4f, AUPR: %.4f, Brier: %.4f' % (AUC, AUPR, Brier))
                ood_res[metric_name] = {'AUC': AUC, 'AUPR': AUPR, 'Brier': Brier}
            else:
                preds = [
                    np.concatenate((val_res, ood_res))
                    # for val_res, ood_res in zip(metric_res['test1'], metric_res['ood'])
                    for val_res, ood_res in zip(metric_res['val'], metric_res['ood'])
                ]
                for i, pred in enumerate(preds):
                    AUC = self.common_get_auc(oracle, pred) # AUC
                    AUPR = self.common_get_aupr(oracle, pred) # AUPR
                    Brier = self.common_get_brier(oracle, pred) # Brier score
                    print('(method %d) AUC: %.4f, AUPR: %.4f, Brier: %.4f' % (i+1, AUC, AUPR, Brier))

                ood_res[metric_name] = [
                    {
                        'AUC': self.common_get_auc(oracle, pred), 
                        'AUPR': self.common_get_aupr(oracle, pred), 
                        'Brier': self.common_get_brier(oracle, pred),
                    } for pred in preds
                ]
      
        # save evaluation res
        if not os.path.exists(os.path.join(self.save_dir, self.task)):
            os.makedirs(os.path.join(self.save_dir, self.task))
        save_name = os.path.join(self.save_dir, self.task, 'uncertainty_ood_eval.res')
        torch.save(ood_res, save_name)



if __name__ == "__main__":

    SHIFT = 'different_time/java_project'
    # SHIFT = 'different_project/java_project'
    # SHIFT = 'different_author/elasticsearch'
    # MODEL = 'code2vec'
    MODEL = 'word2vec'
    # MODEL = 'lstm'
    # MODEL = 'codebert'
    # TASK = 'CodeSummary_Module'
    TASK = 'CodeCompletion_Module'


    eval_m = Uncertainty_Eval(
        res_dir='Uncertainty_Results_new/{}/{}'.format(SHIFT, MODEL),
        save_dir='Uncertainty_Eval/{}/{}'.format(SHIFT, MODEL), 
        task=TASK,
        shift=True,
        # ood=False, # True if ood is evaluated in Eval_res
        ood=True,
    )
    # error/success prediction
    # eval_m.evaluation()

    # ood detection
    eval_m.ood_detect()
