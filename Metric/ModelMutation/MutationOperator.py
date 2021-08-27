import torch
import copy
import numpy as np
import logging
import torch.nn as nn
import math
from BasicalClass import common_predict, common_ten2numpy
from abc import ABCMeta, abstractmethod


class BasicMutation:
    __metaclass__ = ABCMeta

    def __init__(self, model, device, r, acc_tolerant, verbose):
        self.model = model
        self.ration = r
        self.device = device
        self.acc_tolerant = acc_tolerant
        self.verbose = verbose

    @abstractmethod
    def get_mutate_model(self):
        return None, None

    def run(self, data_loader, iter_time, module_id):
        res = []
        while len(res) <= iter_time:
            print('this is the %d model' % (len(res)))
            mutate_model, is_fail = self.get_mutate_model()
            if not is_fail:
                _, pred_y, _ = common_predict(data_loader, mutate_model, self.device, 
                                              module_id=module_id)
                res.append(common_ten2numpy(pred_y).reshape([-1, 1]))
        return np.concatenate(res, axis=1)


class GaussianFuzzing(BasicMutation):
    def __init__(self, model, device, r=0.005, acc_tolerant=0.90, verbose=False):
        super(GaussianFuzzing, self).__init__(model, device, r, acc_tolerant, verbose)
        self.name = 'GaussianFuzzing'

    def get_mutate_model(self):
        try:
            mutation_model = copy.deepcopy(self.model)
            num_weights = 0
            num_layers = 0  # including the bias
            std_layers = []  # store each the standard deviation of paramets of a layer
            for param in mutation_model.parameters():
                num_weights += (param.data.view(-1)).size()[0]
                num_layers += 1
                std_layers.append(param.data.std().item())

            indices = np.random.choice(num_weights, int(num_weights * self.ration), replace=False)
            logging.info('{}/{} weights to be fuzzed'.format(len(indices), num_weights))
            weights_count = 0
            for idx_layer, param in enumerate(mutation_model.parameters()):
                shape = param.data.size()
                num_weights_layer = (param.data.view(-1)).size()[0]
                mutated_indices = set(indices) & set(
                    np.arange(weights_count, weights_count + num_weights_layer))

                if mutated_indices:
                    mutated_indices = np.array(list(mutated_indices))
                    #########################
                    # project the global index to the index of current layer
                    #########################
                    mutated_indices = mutated_indices - weights_count

                    current_weights = param.data.cpu().view(-1).numpy()
                    #####################
                    #  Note: there is a slight difference from the original paper,in which a new
                    #  value is generated via Gaussian distribution with the original single weight as the expectation,
                    #  while we use the mean of all potential mutated weights as the expectation considering the time-consuming.
                    #  In a nut shell, we yield all the mutated weights at once instead of one by one
                    #########################
                    avg_weights = np.mean(current_weights)
                    current_std = std_layers[idx_layer]
                    mutated_weights = np.random.normal(avg_weights, current_std, mutated_indices.size)
                    current_weights[mutated_indices] = mutated_weights
                    new_weights = torch.tensor(current_weights).reshape(shape)
                    param.data = new_weights.to(self.device)
                if self.verbose:
                    print(">>:mutated weights in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_indices),
                                                                             num_weights_layer))
                weights_count += num_weights_layer
        except:
            return None, True
        return mutation_model, False


class WeightShuffling(BasicMutation):
    def __init__(self, model, device, r=0.005, acc_tolerant=0.90, verbose=False):
        super(WeightShuffling, self).__init__(model, device, r, acc_tolerant, verbose)
        self.name = 'WeightShuffling'

    def get_mutate_model(self):
        try:
            unique_neurons = 0
            mutation_model = copy.deepcopy(self.model)
            ####################
            # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
            # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
            ####################
            for param in mutation_model.parameters():
                shape = param.size()
                dim = len(shape)
                if dim > 1:
                    unique_neurons += shape[0]

            indices = np.random.choice(unique_neurons, int(unique_neurons * self.ration), replace=False)
            logging.info('{}/{} weights to be shuffle'.format(len(indices), unique_neurons))
            neurons_count = 0
            for idx_layer, param in enumerate(mutation_model.parameters()):
                shape = param.size()
                dim = len(shape)
                # skip the bias
                if dim > 1:
                    unique_neurons_layer = shape[0]
                    mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
                    if mutated_neurons:
                        mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                        for neuron in mutated_neurons:
                            ori_shape = param.data[neuron].size()
                            old_data = param.data[neuron].view(-1).cpu().numpy()
                            # shuffle
                            shuffle_idx = np.arange(len(old_data))
                            np.random.shuffle(shuffle_idx)
                            new_data = old_data[shuffle_idx]
                            new_data = torch.tensor(new_data).reshape(ori_shape)
                            param.data[neuron] = new_data.to(self.device)
                    if self.verbose:
                        print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                                 unique_neurons_layer))
                    neurons_count += unique_neurons_layer
        except:
            return None, True
        return mutation_model, False


class NeuronSwitch(BasicMutation):
    def __init__(self, model, device, r=0.005, acc_tolerant=0.90, verbose=False):
        super(NeuronSwitch, self).__init__(model, device, r, acc_tolerant, verbose)
        self.name = 'NeuronSwitch'

    def get_mutate_model(self, skip=10):
        try:
            unique_neurons = 0
            mutation_model = copy.deepcopy(self.model)
            ####################
            # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
            # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
            ####################
            for idx_layer, param in enumerate(mutation_model.parameters()):
                shape = param.size()
                dim = len(shape)
                unique_neurons_layer = shape[0]
                # skip the bias
                if dim > 1 and unique_neurons_layer >= skip:
                    import math
                    temp = unique_neurons_layer * self.ration
                    num_mutated = math.floor(temp) if temp > 2. else math.ceil(temp)
                    mutated_neurons = np.random.choice(unique_neurons_layer,
                                                       int(num_mutated), replace=False)
                    switch = copy.copy(mutated_neurons)
                    np.random.shuffle(switch)
                    param.data[mutated_neurons] = param.data[switch]
                    if self.verbose:
                        print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                                 unique_neurons_layer))
        except:
            return None, True
        return mutation_model, False


class NeuronActivationInverse(BasicMutation):
    def __init__(self, model, device, r=0.005, acc_tolerant=0.90, verbose=False):
        super(NeuronActivationInverse, self).__init__(model, device, r, acc_tolerant, verbose)
        self.name = 'NeuronActivationInverse'

    def get_mutate_model(self, act_type='relu'):
        try:
            model = copy.deepcopy(self.model)
            model.train()
            return model, False
            ActFun = nn.ReLU if act_type == 'relu' else nn.ELU
            num_actlayers = 0
            for module in model.modules():
                if isinstance(module, ActFun):
                    num_actlayers += 1
            if num_actlayers == 0:
                raise Exception('No [{}] layer found'.format(ActFun))
            temp = num_actlayers * self.ration
            num_remove = 1 if temp < 1 else math.floor(temp)
            num_remove = int(num_remove)
            idces_remove = np.random.choice(num_actlayers, num_remove, replace=False)
            if self.verbose:
                print('>>>>>>>idces_remove:{}'.format(idces_remove))
            idx = 0
            for name, module in model.named_children():
                # outer relu
                if isinstance(module, nn.ReLU):
                    if idx in idces_remove:
                        model.__delattr__(name)
                        model.__setattr__(name, None)
                    idx += 1
                else:
                    for grand_name, child in module.named_children():
                        if isinstance(child, nn.ReLU):
                            if idx in idces_remove:
                                module.__delattr__(grand_name)
                                model.__setattr__(grand_name, None)
                            idx += 1
        except:
            return None, True
        return model, False
