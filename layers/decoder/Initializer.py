#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch
import torch.nn as nn
import torch.nn.functional as F

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.basic import Linear

class Mean_One(nn.Module):
    def __init__(self, options):
        super(Mean_One, self).__init__()
        self.layer = Linear(options)
    
    def forward(self, h_e, source_mask):
        hidden = torch.sum(h_e * source_mask[:,:,None], dim = 1) / torch.sum(source_mask, dim = 1)[:,None]
        return F.tanh(self.layer(hidden))
    
class Mean_Two(nn.Module):
    def __init__(self, options):
        super(Mean_Two, self).__init__()
        self.layer_1 = Linear(options)
        self.layer_2 = Linear(options)
        
    def forward(self, h_e, source_mask):
        hidden = torch.sum(h_e * source_mask[:,:,None], dim = 1) / torch.sum(source_mask, dim = 1)[:,None]
        return F.tanh(self.layer_1(hidden)), F.tanh(self.layer_2(hidden))

class Last_One(nn.Module):
    def __init__(self, options):
        super(Last_One, self).__init__()
        self.birectional = options['birectional']
        self.n_dim = options['n_in']
        self.layer = Linear(options)
    
    def forward(self, h_e, source_mask):
        if self.birectional:
            hidden_forward = h_e[:,-1,:self.n_dim/2]
            hidden_backward = h_e[:,0,self.n_dim/2:]
            hidden = torch.cat([hidden_forward, hidden_backward], dim = 2)
        else:
            hidden = h_e[:,-1,:]
        return F.tanh(self.layer(hidden))

class Last_Two(nn.Module):
    def __init__(self, options):
        super(Last_Two, self).__init__()
        self.birectional = options['birectional']
        self.n_dim = options['n_in']
        self.layer_1 = Linear(options)
        self.layer_2 = Linear(options)
    
    def forward(self, h_e, source_mask):
        if self.birectional:
            hidden_forward = h_e[:,-1,:self.n_dim/2]
            hidden_backward = h_e[:,0,self.n_dim/2:]
            hidden = torch.cat([hidden_forward, hidden_backward], dim = 2)
        else:
            hidden = h_e[:,-1,:]
        return F.tanh(self.layer_1(hidden)), F.tanh(self.layer_1(hidden))
    
class Initializer(nn.Module):
    def __init__(self, options):
        super(Initializer, self).__init__()
        self.layer = eval(options['type'])(options)
        
    def forward(self, *argv, **kwargs):
        return self.layer(*argv, **kwargs)