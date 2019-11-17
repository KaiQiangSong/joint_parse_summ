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


class LSTM(nn.Module):
    def __init__(self, options, weights = None):
        super(LSTM, self).__init__()
        
        self.n_in = options['n_in']
        self.n_out = options['n_out']
        self.n_layer = options['n_layer']
        self.dropout = options['dropout']
        self.bidirectional = options['bidirectional']
        
        self.layer = nn.LSTM(input_size = self.n_in,
                             hidden_size = self.n_out,
                             num_layers = self.n_layer,
                             dropout = self.dropout,
                             bidirectional = self.bidirectional,
                             batch_first = True
                            )
        
        if weights is not None:
            self.layer.weight = nn.Parameter(weights)
        
        self.fixed = options['fixed']
        if self.fixed:
            self.layer.weight.requires_grad = False
        
    def forward(self, input, hidden_prev = None):
        batch_size = input.batch_sizes[0]
        if hidden_prev is None:
            hidden_prev = (None, None)
        ways = 1
        if self.bidirectional:
            ways = 2
        if hidden_prev[0] is None:
            h_0 = Variable(torch.zeros(self.n_layer * ways, batch_size, self.n_out), requires_grad = False)
        else:
            h_0 = hidden_prev[0]
            h_0 = h_0.repeat(self.n_layer * ways, 1, 1)
            
        if hidden_prev[1] is None:
            c_0 = Variable(torch.zeros(self.n_layer * ways, batch_size, self.n_out), requires_grad = False)
        else:
            c_0 = hidden_prev[1]
            c_0 = c_0.repeat(self.n_layer * ways, 1, 1)
        
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        
        hidden_prev = (h_0, c_0)
        
        return self.layer(input, hidden_prev)
    