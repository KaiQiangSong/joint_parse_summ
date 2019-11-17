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
from layers.data_process import OpMapping_De as OpMapping
from layers.data_process import ActionPool_De as ActionPool
from layers.Loss import calcLossPair


class LSTM2_MeanDiff_FlatParse(nn.Module):
    ROOT, REDUCE_L, REDUCE_R, GEN = 0, 1, 2, 3
    BOS = ROOT
    def __init__(self, options, log, emb_tok_init = None):
        super(LSTM2_MeanDiff_FlatParse, self).__init__()
        self.log = log
        self.options = options
        self.LVT = None
        self.Emb = Embedding(options['network']['Embedding'], emb_tok_init)
        self.Reader = Reader(options['network']['Reader'])
        self.Generator = Generator(options['network']['Generator'])
        
    
    def setLVT(self, LVT):
        self.LVT = LVT
        
    def forward(self, source, len_source, action = None, len_action = None, bi_in = None, debugMode = False, sysGen = None):
        if debugMode:
            source_emb = self.Emb(source)
            h_e, source_mask = self.Reader(source_emb, len_source)
            
            action_emb = self.Emb(action)
            operation = OpMapping.OPITensor2ACTTenso(action)
            outputs_1, _ = self.Generator(self.LVT, (h_e, source_mask), [operation, action_emb, len_action])
            
            Searcher = Debug(self.options)
            Searcher.setType(LSTM2_MeanDiff_FlatParse)
            Searcher.Emb_Set(self.Emb)
            Searcher.Generator_Set(self.Generator)
            Searcher.LVT_Set(self.LVT)
            
            outputs_2 = Searcher.search((h_e, source_mask), action.view(-1), len_action[0])
            return outputs_1, outputs_2
            
        source_emb = self.Emb(source)
        h_e, source_mask = self.Reader(source_emb, len_source)
        
        if action is not None:
            action_emb = self.Emb(action)
            operation = OpMapping.OPITensor2ACTTensor(action)
            outputs, states_next = self.Generator(self.LVT, (h_e, source_mask), [operation, action_emb, len_action])
            
            return outputs, states_next
        
        if sysGen is None:
            Searcher = Searcher_(self.options)
            Searcher.setType(LSTM2_MeanDiff_FlatParse)
            Searcher.Emb_Set(self.Emb)
            Searcher.Generator_Set(self.Generator)
            Searcher.LVT_Set(self.LVT)
            return Searcher.search((h_e, source_mask), bi_in)
        else:
            Searcher = Prob(self.options)
            Searcher.setType(LSTM2_MeanDiff_FlatParse)
            Searcher.Emb_Set(self.Emb)
            Searcher.Generator_Set(self.Generator)
            Searcher.LVT_Set(self.LVT)
            return Searcher.search((h_e, source_mask), sysGen)
    
    def getLoss(self, source, target, sfeat, rfeat):
        '''
            Decoder Input: shift_target_OPI;
            Decoder Output: sharped_target_OPI
        '''
        
        operations = [it['action_map'] for it in rfeat]
        
        source_OPI = OpMapping.TOKSeqs2OPISeqs(source)
        LVT = ActionPool(source_OPI, self.options)
        self.setLVT(LVT)
        padded_source_OPI, len_source = Padding.padding(source_OPI)
    
        target_OPI = OpMapping.OPPSeqs2OPISeqs(operations)
        padded_target_OPI, len_target = Padding.padding(target_OPI)
        # ADD ROOT Operation Before the Sequence
        shifted_target_OPI = Padding.rightShiftCut(padded_target_OPI, OpMapping.ROOT)
        
        sharped_OPI = LVT.sharp(target_OPI)
        padded_sharped_target_OPI, len_target = Padding.padding(sharped_OPI)
        
        outputs, states_pass = self.forward(padded_source_OPI, len_source, shifted_target_OPI, len_target)
        preds_ACT = outputs[0]
        preds_TOK = outputs[1]
        
        mask = (padded_target_OPI >= 3).float()
        
        goldstandard_ACT = OpMapping.OPITensor2ACTTensor(padded_sharped_target_OPI)
        goldstandard_TOK = OpMapping.OPITensor2TOKTensor(padded_sharped_target_OPI)
        # Updating
        loss_ACT = calcLossPair(preds_ACT, goldstandard_ACT, len_target)
        loss_TOK = calcLossPair(preds_TOK, goldstandard_TOK, len_target, mask_ = mask)
        
        return (loss_ACT[0] + loss_TOK[0]) / (loss_ACT[1])
    
    def getAnswer(self, Answers, Vocab, doc):
        text = translate([[tok[0]-3, tok[1]] for tok in Answers if tok[0] >= 3], Vocab, doc)
        parses = translateParse_De(Answers, Vocab, doc)
        return text, parses
        
    def genSummary(self, source, Vocab, doc, bi_in = None):
        # Processing Data
        source_OPI = OpMapping.TOKSeqs2OPISeqs(source)
        LVT = ActionPool(source_OPI, self.options)
        self.setLVT(LVT)
        padded_source_OPI, len_source = Padding.padding(source_OPI)
        
        Answers = self.forward(padded_source_OPI, len_source, bi_in = bi_in)
        #Answers = sorted(Answers, key = lambda x:x[0]/len([it for it in x[1] if it[0] > 2]))
        Answers = sorted(Answers, key = lambda x:x[0]/len(x[1]))
        #print Answers
        #print [it[0] for it in Answers[0]]
        
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
        source_OPI = OpMapping.TOKSeqs2OPISeqs(source)
        LVT = ActionPool(source_OPI, self.options)
        self.setLVT(LVT)
        
        target_OPI = OpMapping.OPPSeqs2OPISeqs([sysGen])
        sharped_OPI = LVT.sharp(target_OPI)[0]
        
        padded_source_OPI, len_source = Padding.padding(source_OPI)
        Answer = self.forward(padded_source_OPI, len_source, sysGen = sharped_OPI)
        return Answer
    
    def debug(self, source, target, sfeat, rfeat):
        operations = [it['oracle_dependency'] for it in rfeat]
        # Processing Data
        source_OPI = OpMapping.TOKSeqs2OPISeqs(source)
        LVT = ActionPool(source_OPI, self.options)
        self.setLVT(LVT)
        padded_source_OPI, len_source = Padding.padding(source_OPI)
    
        target_OPI = OpMapping.OPPSeqs2OPISeqs(operations)
        target_OPI = LVT.sharp(target_OPI)
        
        padded_target_OPI, len_target = Padding.padding(target_OPI)
        shifted_target_OPI = Padding.rightShiftCut(padded_target_OPI, OTHER)
        
        mask = (padded_target_OPI >= 3).float()
        padded_target_ACT = OpMapping.OPITensor2ACTTensor(padded_target_OPI)
        padded_target_TOK = OpMapping.OPITensor2TOKTensor(padded_target_OPI)
        
        outputs_1, outputs_2 = self.forward(padded_source_OPI, len_source, shifted_target_OPI, len_target, debugMode = True)
        
        preds_ACT_1 = outputs_1[0]
        preds_TOK_1 = outputs_1[1]
        
        # Updating
        loss_ACT = calcLossPair(preds_ACT_1, padded_target_ACT, len_target)
        loss_TOK = calcLossPair(preds_TOK_1, padded_target_TOK, len_target, mask_ = mask)
        
        loss_1 = (loss_ACT[0] + loss_TOK[0]) / (loss_ACT[1])
        
        preds_ACT_2 = outputs_2[0]
        preds_TOK_2 = outputs_2[1]
        # Updating
        loss_ACT = calcLossPair(preds_ACT_2, padded_target_ACT, len_target)
        loss_TOK = calcLossPair(preds_TOK_2, padded_target_TOK, len_target, mask_ = mask)
        
        loss_2 = (loss_ACT[0] + loss_TOK[0]) / (loss_ACT[1])
        
        diff = torch.abs(outputs_1[3] - outputs_2[3])[0]
        
        
        n = diff.size()[0]
        m = diff.size()[1]
        '''
        print n, m
        for i in range(n):
            for j in range(m):
                if float(diff[i][j])>1e-4:
                    print i,j, float(diff[i][j])
        '''
        return loss_1, loss_2
    
    @staticmethod
    def state_init():
        '''
            stack:
            top:
            n_gen:
            h_d_init
                torch<->numpy
            c_d_init
                torch<->numpy
        '''
        return [0, 0, (None, None)]
    
    @staticmethod
    def state_pack(s):
        return [s[0], s[1], (numpy2torch(s[2][0]), numpy2torch(s[2][1]))]
    
    @staticmethod
    def state_unpack(s):
        return [s[0], s[1], (torch2numpy(s[2][0]), torch2numpy(s[2][1]))]
    
    @staticmethod
    def state_process(state_old, state_new, action):
        return state_new
    
    @staticmethod
    def cond(preds, state_pass, conditions = None):
        top = state_pass[0]
        n_gen = state_pass[1]
        cands = []
        # ROOT ... X, Y
        if top > 2:
            cands.append(OpMapping.REDUCE_L)
        # ROOT ... X, Y  or ROOT X
        if top > 1:
            cands.append(OpMapping.REDUCE_R)
        # No more than maxLength
        if n_gen < conditions['maxGen']:
            cands += range(3, conditions['maxSize'])
        
        return np.asarray(cands, dtype = np.int64)
    
    @staticmethod
    def getPred(outputs):
        pred_act = outputs[0]
        pred_tok = outputs[1]
        pred = np.concatenate([pred_act[:-1], pred_act[-1] * pred_tok])
        return pred
    
    @staticmethod
    def getAtt(outputs):
        return outputs[2]
    
    @staticmethod
    def lastToken(seq, conditions):
        for item in reversed(seq):
            if item[0] >= 3:
                return item[0]
        return 0
    
    @staticmethod
    def checkEnd(Sequence, State_Pass):
        top = State_Pass[0]
        if (top == 2) and (Sequence[-1][0] == OpMapping.REDUCE_R):
            return True
        return False
    
    @staticmethod
    def getInputs(Token, Lengths, Emb):
        action_emb = Emb(Token)
        operation = OpMapping.OPITensor2ACT4Tensor(Token)
        return [operation, action_emb, Lengths]