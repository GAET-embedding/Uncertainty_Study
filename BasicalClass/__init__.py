from .BasicModule import BasicModule
from .AndroidMalware import Android_Module
from .Cifar10 import CIFAR10_Module
from .Cifar100 import CIFAR100_Module
from .Fashion import Fashion_Module
from .CodeSummary import CodeSummary_Module
from .CodeCompletion import CodeCompletion_Module
from .common_function import *


'''
    this dirotory contains different Module and some common use API
    a Module is a object contains the model, training data, testing data etc.
    module list:
        
    
'''

MODULE_LIST = [
    Fashion_Module,
    CIFAR10_Module,
    CIFAR100_Module,
    Android_Module,
    CodeSummary_Module,
    CodeCompletion_Module,
]
