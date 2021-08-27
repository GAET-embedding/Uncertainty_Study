from BasicalClass import *
from Metric import BasicUncertainty
from Metric.ModelMutation.MutationOperator import GaussianFuzzing, NeuronActivationInverse, WeightShuffling, NeuronSwitch


class Mutation(BasicUncertainty):
    name_list = [
        'GaussianFuzzing',
        'WeightShuffling',
        'NeuronSwitch',
        'NeuronActivationInverse',
    ]

    def __init__(self, instance: BasicModule,  device, iter_time):
        super(Mutation, self).__init__(instance, device)
        self.iter_time = 2 if IS_DEBUG else iter_time
        self.gf = GaussianFuzzing(self.instance.model, device=self.device)
        self.nai = NeuronActivationInverse(self.instance.model, device=self.device)
        self.ws = WeightShuffling(self.instance.model, device=self.device)
        self.ns = NeuronSwitch(self.instance.model, device=self.device)
        self.op_list = [self.gf, self.nai, self.ws, self.ns]

    @staticmethod
    def label_chgrate(orig_pred, prediction):
        _, repeat_num = np.shape(prediction)
        tmp = np.tile(orig_pred.reshape([-1, 1]), (1, repeat_num))
        return np.sum(tmp == prediction, axis=1, dtype=np.float) / repeat_num

    def _uncertainty_calculate(self, data_loader):
            score_list = []
            _, orig_pred, _ = common_predict(
                data_loader, self.model, self.device,
                module_id=self.module_id
            )
            orig_pred = common_ten2numpy(orig_pred)
            for op in self.op_list:
                print(op.__class__.__name__)
                mutation_matrix = op.run(data_loader, iter_time=self.iter_time, module_id=self.module_id)
                score = self.label_chgrate(orig_pred, mutation_matrix)
                score_list.append(score)
            return score_list
