#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch.nn as nn

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.decoder import Decoder 

class Generator_Flat(nn.Module):
    def __init__(self, options):
        super(Generator_Flat, self).__init__()
        self.Decoder = Decoder(options['Decoder'])
        
    def forward(self, LVT, state_below, Inputs, hidden_prev = None):
        target_emb, len_target = Inputs
        return self.Decoder(LVT, target_emb, len_target, state_below, hidden_prev)
