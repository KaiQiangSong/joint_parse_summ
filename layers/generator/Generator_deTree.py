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

class Generator_deTree(nn.Module):
    def __init__(self, options):
        super(Generator_deTree, self).__init__()
        self.node = TreeNode(options['TreeNode'])
        self.Decoder = Decoder(options['Decoder'])
    
    @staticmethod
    def state_init():
        return [[None],[],[],[],-1,-1,0]
    
    def forward(self, LVT, state_below, inputs, state_prev = None):
        '''
            states_prev:
                hidden: current LSTM hiddens for nodes in stack
                stack: current nodes in stack
                pointer: current pointer positions for nodes in stack
                prev: the parent pointers
                top: the top index for stack
                pp: the parent index for top node in stack
                nt: the number of not finished non-terminals
        '''
        operation, action_emb, len_action = inputs
        action_mask = torch.where(operation == 3, torch.ones_like(operation), torch.zeros_like(operation)).float()
        if state_prev is None:
            G = Graph.buildGraphs(self.node, operation, action_emb, len_action)
            stack_emb, len_stacks = G.calc()
            #print stack_emb.size(), len_stacks
            outputs, states_pass = self.Decoder(LVT, state_below, self.node, stack_emb = stack_emb, len_stacks = len_stacks, action_emb = action_emb, action_mask = action_mask, len_action = len_action)
        else:
            outputs, states_pass = self.Decoder(LVT, state_below, self.node, operation = operation, action_emb = action_emb, action_mask = action_mask, len_action = len_action, state_prev = state_prev)
        
        return outputs, states_pass