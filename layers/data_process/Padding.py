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


class Padding(object):
    @staticmethod
    def rightEOS(seqs, patch = 2):
        return [seq + [patch] for seq in seqs]
    
    @staticmethod
    def padding(seqs):
        lengths = [len(seq) for seq in seqs]
        maxLength = max(lengths)
        result = [seqs[i] + [0] * (maxLength - lengths[i]) for i in range(len(seqs))]
        ret1 = Variable(torch.LongTensor(result), requires_grad = False)
        ret2 = Variable(torch.LongTensor(lengths), requires_grad = False)
        #print ret1
        #print ret2
        if torch.cuda.is_available():
            ret1 = ret1.cuda()
            ret2 = ret2.cuda()
        #print ret1
        #print ret2
        return ret1, ret2
    
    @staticmethod
    def padding_withEOS(seqs, patch = 2):
        lengths = [len(seq)+1 for seq in seqs]
        maxLength = max(lengths)
        result = [seqs[i] + [2] + [0] * (maxLength - lengths[i]) for i in range(len(seqs))]
        ret1 = Variable(torch.LongTensor(result), requires_grad = False)
        ret2 = Variable(torch.LongTensor(lengths), requires_grad = False)
        #print ret1
        #print ret2
        if torch.cuda.is_available():
            ret1 = ret1.cuda()
            ret2 = ret2.cuda()
        #print ret1
        #print ret2
        return ret1, ret2
    
    @staticmethod
    def unpadding(seqs, lengths):
        batch_size = lengths.size()[0] 
        return [[int(seqs[i][j]) for j in range(lengths[i])]for i in range(batch_size)]
    
    @staticmethod
    def rightShift(x, patch = 1):
        res = F.pad(x, (1, 0), "constant", patch)
        return res
    
    @staticmethod
    def rightShiftCut(x, patch = 1):
        res = F.pad(x, (1, 0), "constant", patch)[:,:-1]
        return res
    
    @staticmethod
    def rightPadEnd(x, patch = 2):
        res = F.pad(x, (0, 1), "constant", patch)
        return res