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

#==============================#
#    Class/Layer Part Import   #
#==============================#


class Linear(nn.Module):
    def __init__(self, options, weights = None):
        super(Linear, self).__init__()
        
        self.n_in = options['n_in']
        self.n_out = options['n_out']
        
        self.layer = nn.Linear(self.n_in, self.n_out)
        
        if weights is not None:
            self.layer.weight = nn.Parameter(self.load(weights))
        else:
            nn.init.xavier_normal_(self.layer.weight)
            
        self.fixed = options['fixed']
        if self.fixed:
            self.layer.weight.requires_grad = False
        
    def forward(self, input):
        return self.layer(input)
