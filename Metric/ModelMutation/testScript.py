import argparse
from Metric.ModelMutation.MutationMethod import Mutation
from BasicalClass import *



def main(data_type, device_id, is_poor):
    device = torch.device("cuda:"+ str(device_id) if torch.cuda.is_available() else "cpu")
    module = data_type(device= device, load_poor=is_poor)
    print('now the module type is', module.__class__.__name__)
    m = Mutation(module, device=device, it_time = 500)
    m.evaluate()   # run evaluate and get the auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--data_type', type=int, default=1, help='the module type')
    parser.add_argument('--device_id', type=int, default=0, help='gpu id')
    parser.add_argument('--is_poor', type=int, default=0, help='load poor model')
    args = parser.parse_args()

    main(MODULE_LIST[args.data_type], args.device_id, args.is_poor)
