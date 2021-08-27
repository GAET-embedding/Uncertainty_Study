import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from BasicalClass import common_predict, common_get_maxpos
from BasicalClass import BasicModule
from Metric import BasicUncertainty


class ModelWithTemperature(BasicUncertainty):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, instance: BasicModule, device, temperature=None):
        super(ModelWithTemperature, self).__init__(instance, device)
        if temperature is None:
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            self.set_temperature(self.val_loader)
        else:
            self.temperature = temperature
        torch.save(self.temperature, os.path.join(
            instance.save_dir, instance.__class__.__name__, 'temperature.tmp'
        ))

    # modified forward for code summary task
    def forward(self, *input, **kwargs):
        # here since the model is code_summary model, the input has to be changed
        logits = self.model(*input, **kwargs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # self.to(self.device)
        # nll_criterion = nn.CrossEntropyLoss().cuda()
        # ece_criterion = _ECELoss().cuda()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():

            if self.module_id == 0: # code summary
                for i, ((sts, paths, eds), y, length) in enumerate(valid_loader):
                    torch.cuda.empty_cache()
                    sts = sts.to(self.device)
                    paths = paths.to(self.device)
                    eds = eds.to(self.device)
                    y = torch.tensor(y, dtype=torch.long)
                    logits = self.model(starts=sts, paths=paths, ends=eds, length=length)
                    
                    # detach
                    sts = sts.detach().cpu()
                    paths = paths.detach().cpu()
                    eds = eds.detach().cpu()

                    if isinstance(logits, tuple):
                        logits = (py.detach().cpu() for py in logits)
                    else:
                        logits = logits.detach().cpu()

                    logits_list.append(logits)
                    labels_list.append(y)

            elif self.module_id == 1: # code completion
                for i, (input, y, _) in enumerate(valid_loader):
                    torch.cuda.empty_cache()
                    input = input.to(self.device)
                    logits = self.model(input)

                    # detach
                    input = input.detach().cpu()
                    logits = logits.detach().cpu()
                
                    logits_list.append(logits)
                    labels_list.append(y.long())
            else:
                raise TypeError()


            # logits = torch.cat(logits_list).to(self.device)
            # labels = torch.cat(labels_list).to(self.device)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        return self

    def _uncertainty_calculate(self, data_loader):
        score, _, _ = common_predict(data_loader, self, self.device, module_id=self.module_id)
        score = common_get_maxpos(score)
        return score


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece