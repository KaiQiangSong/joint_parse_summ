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


class BFS_BEAM(BasicSearcher):
    def __init__(self, options):
        super(BFS_BEAM, self).__init__(options)
    
    def search(self, state_below, bi_in = None):
        '''
            States:
                Score:
                Decoded Sequences:
                State_Pass:
                biGrams:
            Outputs:
                predictions:
                ...
        '''
        startState = [0, [[self.BOS, None]], None, None]
        Candidates = [[startState]]
        pool = np.asarray(self.LVT.Pool, dtype = np.int64)
        maxSize = len(self.LVT.Pool)
        self.condition_new('maxSize',maxSize)
        Answers = []
        
        for l in range(self.maxLength):
            cands = Candidates[l]
            
            topK = []
            for cand in cands:
                Score, Sequence, State_Pass, bi_old = cand
                
                if State_Pass is None:
                    State_Pass = self.state_init()
                #print Score, [it[0] for it in Sequence],
                    
                State_packed = self.state_pack(State_Pass)
                
                token = Sequence[-1][0]
                Token = Variable(torch.LongTensor([[token]]))
                Length = Variable(torch.LongTensor([1]))
                
                if torch.cuda.is_available():
                    Token = Token.cuda()
                    Length = Length.cuda()
                
                '''
                Token_Emb = self.Emb(Token)
                outputs, state_pass_ = self.generator(self.LVT, state_below, Token_Emb, Length, State_packed)
                '''
                
                
                Inputs = self.getInputs(Token, Length)
                outputs, state_pass_ = self.generator(self.LVT, state_below, Inputs, State_packed)
                    
                state_pass = self.state_process(State_packed, state_pass_, token)
                state_unpacked = self.state_unpack(state_pass)
                #print state_unpacked[1], state_unpacked[2]
                outputs = BasicSearcher.atom(outputs)
                preds = self.getPred(outputs)
                preds = np.log(preds + 1e-8)
                
                # Conditions
                ids = self.cond(preds, state_unpacked, self.conditions)
                # biGramTrick
                if self.biGramTrick:
                    lastToken = self.lastToken(Sequence, self.conditions)
                    if bi_old is None:
                        bi_old = set()
                    preds, ids, bi_next = self.do_bigramTrick(preds, ids, self.beam_size, lastToken, bi_in, bi_old, self.gamma, pool)
                    topK_i = [[-preds[i]+Score, Sequence+[[int(pool[ids[i]]), self.getAtt(outputs)]], copy.deepcopy(state_unpacked), bi_next[i]] for i in range(preds.shape[0])]
                else:
                    preds, ids = self.no_bigramTrick(preds, ids, self.beam_size)
                    topK_i = [[-preds[i]+Score, Sequence+[[int(pool[ids[i]]), self.getAtt(outputs)]], copy.deepcopy(state_unpacked), None] for i in range(preds.shape[0])]
                
                topK += topK_i
                
            topK = sorted(topK, key = lambda x:x[0])[:self.beam_size * 2]
            topK_next = []
            for Score, Sequence, State_Pass, biGram in topK:
                if self.checkEnd(Sequence, State_Pass):
                    Answers.append([Score, Sequence, State_Pass, biGram])
                else:
                    topK_next.append([Score, Sequence, State_Pass, biGram])
            
            if (len(Answers) >= self.beam_size):
                break
            
            if len(topK_next) > self.beam_size:
                topK_next = topK_next[:self.beam_size]
            
            Candidates.append(topK_next)
            
        if len(Answers) < 1:
            return sorted(Candidates[-1], key = lambda x:x[0])
        
        Answers = sorted(Answers, key = lambda x:x[0])
        if (len(Answers) > self.beam_size):
            Answers = Answers[:self.beam_size]
        
        return Answers
