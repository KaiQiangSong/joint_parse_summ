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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.basic import Attention, Linear, LSTM
from .Initializer import Initializer
from layers.utils import oneHot


class Decoder_FlatParse(nn.Module):
    ROOT, REDUCE_L, REDUCE_R, GEN = 0, 1, 2, 3
    def __init__(self, options):
        super(Decoder_FlatParse, self).__init__()
        self.init = Initializer(options['init'])
        self.main = LSTM(options['main'])
        self.predict_act = Linear(options['predict_act'])
        self.attention = Attention(options['attention'])
        self.hidden = Linear(options['hidden'])
        self.predict_tok = Linear(options['predict_tok'])
        self.switcher_copy = Linear(options['switcher_copy'])
    
    def forward(self, LVT, state_below, operation, target_emb, len_target, hidden_prev = None):
        if hidden_prev is not None:
            top, n_gen, hidden_prev = hidden_prev
            operation = operation[0][0]
            oper = int(operation)
            if (oper == self.ROOT):
                top += 1
            elif (oper == self.GEN):
                top += 1
                n_gen += 1
            elif oper == self.REDUCE_L:
                top -= 1
            elif oper == self.REDUCE_R:
                top -= 1
            
        h_e, source_mask = state_below
        
        if (hidden_prev is None) or (hidden_prev[0] is None) or (hidden_prev[1] is None):
            h_d_init, c_d_init = self.init(h_e, source_mask)
        
        sorted_len_target, IDs = torch.sort(len_target, descending = True)
        _, bIDs = torch.sort(IDs, descending = False)
        sorted_target_emb = torch.index_select(target_emb, 0, IDs)
        
        packed_target_emb = pack_padded_sequence(sorted_target_emb, [int(it) for it in list(sorted_len_target)], batch_first = True)
    
        if (hidden_prev is None) or (hidden_prev[0] is None) or (hidden_prev[1] is None):
            h_d_init = torch.index_select(h_d_init, 0, IDs)
            c_d_init = torch.index_select(c_d_init, 0, IDs)
            
        if hidden_prev is None:
            h_d, (ph_d, pc_d) = self.main(packed_target_emb, (h_d_init, c_d_init))
        else:
            if hidden_prev[0] is None:
                hidden_prev = (h_d_init, hidden_prev[1])
            if hidden_prev[1] is None:
                hidden_prev = (hidden_prev[0], c_d_init)
            h_d, (ph_d, pc_d) = self.main(packed_target_emb, hidden_prev)
        
        h_d = pad_packed_sequence(h_d, batch_first=True)
        
        h_d = torch.index_select(h_d[0], 0, bIDs)
        ph_d = torch.index_select(ph_d, 1, bIDs)
        pc_d = torch.index_select(pc_d, 1, bIDs)
        
        pred_act = F.softmax(self.predict_act(h_d), dim = 2)
        
        alpha = self.attention(h_e, h_d, source_mask)
        context = torch.bmm(alpha, h_e)
        
        hidden = F.tanh(self.hidden(torch.cat([h_d, context], dim = 2)))
        pred_tok = F.softmax(self.predict_tok(hidden), dim = 2)
        
        
        switcher_copy = F.sigmoid(self.switcher_copy(torch.cat([h_d, context, target_emb], dim = 2)))
        pool_size = LVT.size
        
        if pool_size > pred_tok.size()[2]:
            if torch.cuda.is_available():
                prob_vocab = switcher_copy * torch.cat([pred_tok, Variable(torch.zeros((pred_tok.size()[0], pred_tok.size()[1], pool_size - pred_tok.size()[2])), requires_grad = False).cuda()], dim = 2)
            else:
                prob_vocab = switcher_copy * torch.cat([pred_tok, Variable(torch.zeros((pred_tok.size()[0], pred_tok.size()[1], pool_size - pred_tok.size()[2])), requires_grad = False)], dim = 2)
        else:
            prob_vocab = switcher_copy * pred
            
        shp = LVT.Pointer.size()
        pointer = oneHot(LVT.Pointer.view(-1), pool_size).view(shp[0], shp[1], pool_size)
        prob_copy = (1 - switcher_copy) * torch.bmm(alpha, pointer)
        prob = prob_vocab + prob_copy
        
        if hidden_prev is not None:
            return [pred_act, prob, alpha], [top, n_gen, (ph_d, pc_d)]
        
        return [pred_act, prob, alpha], []
   