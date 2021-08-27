from torch.nn.modules import module
from Metric import *
from BasicalClass import CodeSummary_Module, CodeCompletion_Module
import torch
from BasicalClass import common_get_auc
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module_id', type=int, default=0, choices=[0,1], 
                        help='the task id, 0 means code summary, 1 means code completion')
    parser.add_argument('--res_dir', type=str, default='program_tasks/code_summary/result/java')
    parser.add_argument('--data_dir', type=str, default='data/java_pkl')
    parser.add_argument('--ood_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='Uncertainty_Results/java')
    parser.add_argument('--load_poor', action='store_true', default=False, 
                        help='use poor pretrained model')
    parser.add_argument('--max_size', type=int, default=None, help='max size of data to be used')
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)

    args = parser.parse_args()
    print(vars(args))

    ModuleList = [
        CodeSummary_Module,
        CodeCompletion_Module,
    ]
    MetricList = [
        Vanilla,
        ModelWithTemperature,
        PVScore,
        ModelActivateDropout,
        Entropy,
        Mutation,
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module_id = args.module_id
    res_dir = args.res_dir
    save_dir = args.save_dir
    data_dir = args.data_dir
    ood_dir = args.ood_dir
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    max_size = args.max_size
    load_poor = args.load_poor
    Module = ModuleList[args.module_id] # code_summary

    module_instance = Module(
        device=device, res_dir=res_dir, save_dir=save_dir, data_dir=data_dir, 
        ood_dir=ood_dir, module_id=module_id, train_batch_size=train_batch_size, 
        test_batch_size=test_batch_size, max_size=max_size, load_poor=load_poor,
    )
    for i, metric in enumerate(MetricList):
        print(f'metric name: {metric.__name__}')
        if metric.__name__ not in ['ModelActivateDropout', 'Mutation']:
            v = metric(module_instance, device)
        else:
            if metric.__name__ == 'Mutation':
                v = metric(module_instance, device, iter_time=10)
            else:
                v = metric(module_instance, device, iter_time=50)
        v.run()