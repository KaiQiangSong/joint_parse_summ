#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.basic import Linear, LSTM
from .Initializer import Initializer

class Decoder_Flat(nn.Module):
    def __init__(self, options):
        super(Decoder_Flat, self).__init__()
        self.init = Initializer(options['init'])
        self.main = LSTM(options['main'])
        self.predict = Linear(options['predict'])
    
    def forward(self, target_emb, len_target, state_below, hidden_prev = None):
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
        
        pred = F.softmax(self.predict(h_d), dim = 2)
        
        return [pred], [ph_d, pc_d]
