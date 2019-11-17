#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch
import torch.nn as nn

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.decoder import Decoder
from layers.graph import Graph_De as Graph
from layers.graph import TreeNode

class Generator_FlatParse(nn.Module):
    def __init__(self, options):
        super(Generator_FlatParse, self).__init__()
        self.Decoder = Decoder(options['Decoder'])
    
    def forward(self, LVT, state_below, inputs, state_prev = None):
        operation, action_emb, len_action = inputs
        if state_prev is None:
            outputs, states_pass = self.Decoder(LVT, state_below, operation, action_emb, len_action)
        else:
            outputs, states_pass = self.Decoder(LVT, state_below, operation, action_emb, len_action, state_prev)
            
        return outputs, states_pass