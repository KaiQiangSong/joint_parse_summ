#==============================#
#    System Import             #
#==============================#


#==============================#
#    Platform Import           #
#==============================#
import numpy as np
import torch
import torch.nn as nn

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.utils import *
from layers.basic import Embedding
from layers.reader import Reader
from layers.generator import Generator
from layers.searcher import Searcher as Searcher_
from layers.searcher import Debug, Prob  
from layers.data_process import Padding
from layers.data_process import LargeVocabularyTrick
from layers.Loss import calcLoss

BOS, EOS = 1, 2

class LSTM2_MeanDiff_FlatToken(nn.Module):
    BOS, EOS = 1, 2
    
    def __init__(self, options, log, emb_tok_init = None):
        super(LSTM2_MeanDiff_FlatToken, self).__init__()
        self.log = log
        self.options = options
        self.LVT = None
        
        self.Emb = Embedding(options['network']['Embedding'])
        self.Reader = Reader(options['network']['Reader'])
        self.Generator = Generator(options['network']['Generator'])
        
    def setLVT(self, LVT):
        self.LVT = LVT
        
    def forward(self, source, len_source, target = None, len_target = None, bi_in = None, debugMode = False, sysGen = None):
        source_emb = self.Emb(source)
        h_e, source_mask = self.Reader(source_emb, len_source)
        
        if target is not None:
            target_emb = self.Emb(target)
            outputs, states_next = self.Generator(self.LVT, (h_e, source_mask), [target_emb, len_target])
            return outputs, states_next
        
        if sysGen is None:
            Searcher = Searcher_(self.options)
            Searcher.setType(LSTM2_MeanDiff_FlatToken)
            Searcher.Emb_Set(self.Emb)
            Searcher.Generator_Set(self.Generator)
            Searcher.LVT_Set(self.LVT)
            return Searcher.search((h_e, source_mask), bi_in)
        else:
            Searcher = Prob(self.options)
            Searcher.setType(LSTM2_MeanDiff_FlatToken)
            Searcher.Emb_Set(self.Emb)
            Searcher.Generator_Set(self.Generator)
            Searcher.LVT_Set(self.LVT)
            return Searcher.search((h_e, source_mask), sysGen)
    
    def getLoss(self, source, target, sfeat, rfeat):
        '''
            Decoder Input: shift_target_OPI;
            Decoder Output: sharped_target_OPI
        '''
        
        LVT = LargeVocabularyTrick(source, self.options)
        self.setLVT(LVT)
        padded_source, len_source = Padding.padding(source)
        
        target = Padding.rightEOS(target)
        padded_target, len_target = Padding.padding(target)
        # ADD BOS Before the Sequence
        shifted_target = Padding.rightShiftCut(padded_target, self.BOS)
        
        sharped_target = LVT.sharp(target)
        padded_sharped_target, len_target = Padding.padding(sharped_target)
        
        outputs, states_pass = self.forward(padded_source, len_source, shifted_target, len_target)
        
        preds = outputs[0]
        loss = calcLoss(preds, padded_sharped_target, len_target)
        return loss
    
    def getAnswer(self, Answers, Vocab, doc):
        text = translate([[tok[0], tok[1]] for tok in Answers], Vocab, doc)
        parses = "No Parses In this Model."
        return text, parses
        
    def genSummary(self, source, Vocab, doc, bi_in = None):
        # Processing Data
        LVT = LargeVocabularyTrick(source, self.options)
        self.setLVT(LVT)
        padded_source, len_source = Padding.padding(source)
        
        Answers = self.forward(padded_source, len_source, bi_in = bi_in)
        Answers = sorted(Answers, key = lambda x:x[0]/len(x[1]))
        
        textAnswers = []
        parseAnswers = []
        N = len(Answers)
        
        for i in range(N):
            text, parse = self.getAnswer(Answers[i][1][1:-1], Vocab, doc)
            textAnswers.append(text)
            parseAnswers.append(parse)
        return [textAnswers, parseAnswers]
    
    def getProb(self, source, Vocab, sysGen):
        # Processing Data
        LVT = ActionPool(source, self.options)
        self.setLVT(LVT)
        
        target = OpMapping.OPPSeqs2OPISeqs([sysGen])
        sharped_target = LVT.sharp(target)[0]
        
        padded_source, len_source = Padding.padding(source)
        Answer = self.forward(padded_source, len_source, sysGen = sharped_target)
        return Answer
    
    def debug(self, source, target, sfeat, rfeat):
        pass
    
    @staticmethod
    def state_init():
        '''
            ph_d, ch_d
        '''
        return [None, None]
    
    @staticmethod
    def state_pack(s):
        return [numpy2torch(s[0]), numpy2torch(s[1])]
    
    @staticmethod
    def state_unpack(s):
        return [torch2numpy(s[0]), torch2numpy(s[1])]
    
    @staticmethod
    def state_process(state_old, state_new, action):
        return state_new
    
    @staticmethod
    def cond(preds, state_pass, conditions = None):
        return  np.asarray(range(conditions['maxSize']))
    
    @staticmethod
    def getPred(outputs):
        return outputs[0]
    
    @staticmethod
    def getAtt(outputs):
        return outputs[1]
    
    @staticmethod
    def lastToken(seq, conditions):
        return seq[-1][0]
    
    @staticmethod
    def checkEnd(Sequence, State_Pass):
        return (Sequence[-1][0] == EOS)
    
    @staticmethod
    def getInputs(Token, Lengths, Emb):
        target_emb = Emb(Token)
        len_target = Lengths
        return [target_emb, len_target]