#==============================#
#    System Import             #
#==============================#


#==============================#
#    Platform Import           #
#==============================#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.utils import *

def crossEntropy(preds, target, mask):
        n_sample = preds.size()[0]
        n_class = preds.size()[1]
        idx = Variable(torch.arange(n_sample, out = torch.LongTensor()), requires_grad = False)
        if torch.cuda.is_available():
            idx = idx.cuda()
        
        cost = -torch.log(torch.index_select(preds.view(-1), 0, n_class *  idx + target) + 1e-8)
        return torch.sum(cost * mask) / torch.sum(mask)
    
def crossEntropyPair(preds, target, mask):
        n_sample = preds.size()[0]
        n_class = preds.size()[1]
        idx = Variable(torch.arange(n_sample, out = torch.LongTensor()), requires_grad = False)
        if torch.cuda.is_available():
            idx = idx.cuda()
        
        cost = -torch.log(torch.index_select(preds.view(-1), 0, n_class *  idx + target) + 1e-8)
        return (torch.sum(cost * mask), torch.sum(mask)) 
    
def calcLoss(preds, target, len_target, mask_ = None):
    n_class = preds.size()[2]
    mask = genMask(len_target)
    if mask_ is not None:
        mask = mask * mask_
    result = crossEntropy(preds.view(-1, n_class), target.view(-1), mask.view(-1))
    return result

def calcLossPair(preds, target, len_target, mask_ = None):
    n_class = preds.size()[2]
    mask = genMask(len_target)
    if mask_ is not None:
        mask = mask * mask_
    result = crossEntropyPair(preds.view(-1, n_class), target.view(-1), mask.view(-1))
    return result