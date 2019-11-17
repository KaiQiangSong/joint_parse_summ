#==============================#
#    System Import             #
#==============================#
import copy

#==============================#
#    Platform Import           #
#==============================#
import numpy as np
import torch
from torch.autograd import Variable

#==============================#
#    Class/Layer Part Import   #
#==============================#
from .BasicSearcher import BasicSearcher

class Prob(BasicSearcher):
    def __init__(self, options):
        super(Prob, self).__init__(options)
    
    def search(self, state_below, sysGen):
        L = len(sysGen)
        startState = [0, [[self.BOS, None]], None, None]
        states = [startState]
        pool = np.asarray(self.LVT.Pool, dtype = np.int64)
        maxSize = len(self.LVT.Pool)
        self.condition_new('maxSize',maxSize)
        for l in range(L):
            current = states[l]
            Score, Sequence, State_Pass, bi_old = current
            if State_Pass is None:
                State_Pass = self.state_init()
            
            State_packed = self.state_pack(State_Pass)
            token = Sequence[-1][0]
            Token = Variable(torch.LongTensor([[token]]))
            Length = Variable(torch.LongTensor([1]))
            
            if torch.cuda.is_available():
                Token = Token.cuda()
                Length = Length.cuda()
                
            Inputs = self.getInputs(Token, Length)
            outputs, state_pass_ = self.generator(self.LVT, state_below, Inputs, State_packed)
            
            state_pass = self.state_process(State_packed, state_pass_, token)
            state_unpacked = self.state_unpack(state_pass)
            
            outputs = BasicSearcher.atom(outputs)
            preds = self.getPred(outputs)
            preds = -np.log(preds + 1e-8)
            
            id = sysGen[l]

            newState = [preds[id] + Score, Sequence + [[int(pool[id]), self.getAtt(outputs)]], copy.deepcopy(state_unpacked), None]
            states.append(newState)
        
        return states[-1][0]