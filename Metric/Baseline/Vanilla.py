from BasicalClass import BasicModule
from BasicalClass import common_get_maxpos, common_predict
from Metric import BasicUncertainty


class Vanilla(BasicUncertainty):
    def __init__(self, instance: BasicModule, device):
        super(Vanilla, self).__init__(instance, device)

    def _uncertainty_calculate(self, data_loader):
        pred_pos, _, _ = common_predict(
            data_loader, self.model, self.device, 
            module_id=self.module_id
        )
        return common_get_maxpos(pred_pos)