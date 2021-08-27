import os
import torch
import numpy as np
from Uncertainty_Eval.evaluation import Uncertainty_Eval

class OODdetect(Uncertainty_Eval):

    def __init__(self, res_dir, projects, save_dir, task='CodeSummary_Module', shift=False):
        super(OODdetect, self).__init__(res_dir, projects, save_dir, task, shift)

    def evaluate(self):
        eval_res = {}
        project = self.projects
        trg_dir = os.path.join(self.res_dir, project, self.task)
        truth = torch.load(os.path.join(trg_dir,'truth.res'))
        uncertainty_res = [
            f for f in os.listdir(trg_dir) if f.endswith('.res') and f != 'truth.res'
        ]
        if not self.shift:
            print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f' % (
                np.mean(truth['train']), np.mean(truth['val']), np.mean(truth['test'])
            ))
            # val as in-distribution, test as out-of-distribution
            oracle = np.array([1]*len(truth['val']) + [0]*len(truth['test']))
        else:
            print('train_acc: %.4f, val_acc: %.4f, test1_acc: %.4f, test2_acc: %.4f, test3_acc: %.4f' % (
                np.mean(truth['train']), np.mean(truth['val']), 
                np.mean(truth['test1']), np.mean(truth['test2']), np.mean(truth['test3'])
            ))
            oracle1 = np.array([1]*len(truth['val']) + [0]*len(truth['test1']))
            oracle2 = np.array([1]*len(truth['val']) + [0]*len(truth['test2']))
            oracle3 = np.array([1]*len(truth['val']) + [0]*len(truth['test3']))
         
        for metric in uncertainty_res:
            metric_res = torch.load(os.path.join(trg_dir, metric))
            metric_name = metric[:-4] # get rid of endswith '.res'
            print('\n%s:' % (metric_name))
            eval_res[metric_name] = {}

            if not self.shift:
                if metric_name not in ['Mutation', 'PVScore']:
                    pred = np.concatenate((metric_res['val'], metric_res['test']))
                else:
                    pred = np.concatenate((metric_res['val'][0], metric_res['test'][0]))

                # AUC
                AUC = self.common_get_auc(oracle, pred)
                print('AUC: %.4f' % (AUC))
                # AUPR
                AUPR = self.common_get_aupr(oracle, pred)
                print('AUPR: %.4f' % (AUPR))
                # Brier score
                Brier = self.common_get_brier(oracle, pred)
                print('Brier: %.4f' % (Brier))

                eval_res[metric_name]['AUC'] = AUC
                eval_res[metric_name]['AUPR'] = AUPR
                eval_res[metric_name]['Brier'] = Brier
            else:
                if metric_name not in ['Mutation', 'PVScore']:
                    pred1 = np.concatenate((metric_res['val'], metric_res['test1']))
                    pred2 = np.concatenate((metric_res['val'], metric_res['test2']))
                    pred3 = np.concatenate((metric_res['val'], metric_res['test3']))
                else:
                    pred1 = np.concatenate((metric_res['val'][0], metric_res['test1'][0]))
                    pred2 = np.concatenate((metric_res['val'][0], metric_res['test2'][0]))
                    pred3 = np.concatenate((metric_res['val'][0], metric_res['test3'][0]))

                # AUC
                AUC1 = self.common_get_auc(oracle1, pred1)
                AUC2 = self.common_get_auc(oracle2, pred2)
                AUC3 = self.common_get_auc(oracle3, pred3)
                print('AUC1: %.4f, AUC2: %.4f, AUC3: %.4f' % (AUC1, AUC2, AUC3))
                # AUPR
                AUPR1 = self.common_get_aupr(oracle1, pred1)
                AUPR2 = self.common_get_aupr(oracle2, pred2)
                AUPR3 = self.common_get_aupr(oracle3, pred3)
                print('AUPR1: %.4f, AUPR2: %.4f, AUPR3: %.4f' % (AUPR1, AUPR2, AUPR3))
                # Brier score
                Brier1 = self.common_get_brier(oracle1, pred1)
                Brier2 = self.common_get_brier(oracle2, pred2)
                Brier3 = self.common_get_brier(oracle3, pred3)
                print('Brier1: %.4f, Brier2: %.4f, Brier3: %.4f' % (Brier1, Brier2, Brier3))

                eval_res[metric_name]['AUC1'] = AUC1
                eval_res[metric_name]['AUPR1'] = AUPR1
                eval_res[metric_name]['Brier1'] = Brier1
                eval_res[metric_name]['AUC2'] = AUC2
                eval_res[metric_name]['AUPR2'] = AUPR2
                eval_res[metric_name]['Brier2'] = Brier2
                eval_res[metric_name]['AUC3'] = AUC3
                eval_res[metric_name]['AUPR3'] = AUPR3
                eval_res[metric_name]['Brier3'] = Brier3

        # save evaluation res
        if not os.path.exists(os.path.join(self.save_dir, self.task)):
            os.makedirs(os.path.join(self.save_dir, self.task))
        save_name = os.path.join(self.save_dir, self.task, 'uncertainty_ood_eval.res')
        torch.save(eval_res, save_name)

            

if __name__ == "__main__":
    # from preprocess.train_split import JAVA_PROJECTS

    # eval_m = Uncertainty_Eval(
    #     res_dir='Uncertainty_Results', projects=JAVA_PROJECTS, 
    #     save_dir='Uncertainty_Eval/java', task='CodeSummary_Module'
    # )
    # eval_m = OODdetect(
    #     res_dir='Uncertainty_Results/different_project', 
    #     projects='java_project3', 
    #     save_dir='Uncertainty_Eval/different_project/java_project3', 
    #     # task='CodeSummary_Module'
    #     task='CodeCompletion_Module',
    # )
    # eval_m.evaluate()

    # eval_m = OODdetect(
    #     res_dir='Uncertainty_Results/different_author', 
    #     projects='elasticsearch', 
    #     save_dir='Uncertainty_Eval/different_author/elasticsearch', 
    #     # task='CodeSummary_Module',
    #     task='CodeCompletion_Module',
    #     shift=True,
    # )
    # eval_m.evaluate()

    eval_m = OODdetect(
        res_dir='Uncertainty_Results/different_time', 
        projects='java_project', 
        save_dir='Uncertainty_Eval/different_time/java_project', 
        # task='CodeSummary_Module',
        task='CodeCompletion_Module',
        shift=True
    )
    eval_m.evaluate()
        
