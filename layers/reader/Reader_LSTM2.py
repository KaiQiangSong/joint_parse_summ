#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.encoder import Encoder
from layers.utils import *

class Reader_LSTM2(nn.Module):
    def __init__(self, options):
        super(Reader_LSTM2, self).__init__()
        self.Encoder = Encoder(options['Encoder'])
        
    def forward(self, source_emb, len_source):
        
        # Sorting with Lengths
        sorted_len_source, IDs = torch.sort(len_source, descending = True)
        _, bIDs = torch.sort(IDs, descending = False)    
        sorted_source_emb = torch.index_select(source_emb, 0, IDs)
        
        # Get the Hiddens
        packed_source_emb = pack_padded_sequence(sorted_source_emb, [int(it) for it in list(sorted_len_source)], batch_first = True)
        h_e, (ph_e, pc_e)  = self.Encoder(packed_source_emb)
        h_e = pad_packed_sequence(h_e, batch_first = True)
        h_e = h_e[0]
        
        # Generate
        source_mask = genMask(sorted_len_source)
        
        h_e = torch.index_select(h_e, 0, bIDs)
        source_mask = torch.index_select(source_mask, 0, bIDs)
        
        return h_e, source_mask