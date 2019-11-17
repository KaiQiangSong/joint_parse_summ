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

class MaskLSTMCell(nn.Module):
    def __init__(self, options):
        super(MaskLSTMCell, self).__init__()
        
        self.n_in = options['n_in']
        self.n_out = options['n_out']
        
        self.input = nn.Linear(self.n_in, self.n_out * 4)
        self.hidden = nn.Linear(self.n_out, self.n_out * 4, bias = False)
    
    def forward(self, input, mask, h_prev, c_prev):
        activation = self.input(input) + self.hidden(h_prev)
        
        activation_i = activation[:,:self.n_out]
        activation_f = activation[:,self.n_out:self.n_out*2]
        activation_c = activation[:,self.n_out*2:self.n_out*3]
        activation_o = activation[:,self.n_out*3:self.n_out*4]

        i = torch.sigmoid(activation_i)
        f = torch.sigmoid(activation_f)
        o = torch.sigmoid(activation_o)
        
        
        c = f * c_prev + i * torch.tanh(activation_c)
        c = mask[:,None] * c + (1-mask)[:,None] * c_prev
        
        h = o * torch.tanh(c)
        h = mask[:,None] * h + (1-mask)[:,None] * h_prev
        
        return h, c
    
class MaskLSTM(nn.Module):
    def __init__(self, options):
        super(MaskLSTM, self).__init__()
        self.n_in = options['n_in']
        self.n_out = options['n_out']
        self.Cell = MaskLSTMCell(options)
    
    def forward(self, inputs, masks, prev_state = None):
        batch_size = int(inputs.size()[0])
        maxLength = int(inputs.size()[1])
        
        if prev_state is None:
            h_prev = Variable(torch.zeros(batch_size, self.n_out), requires_grad = False)
            c_prev = Variable(torch.zeros(batch_size, self.n_out), requires_grad = False)
            if torch.cuda.is_available():
                h_prev = h_prev.cuda()
                c_prev = c_prev.cuda()
        else:
            h_prev, c_prev = prev_state
        
        h = []
        c = []
        for i in range(maxLength):
            input = inputs[:,i,:]
            mask = masks[:,i]
            h_prev, c_prev = self.Cell(input, mask, h_prev, c_prev)
            h.append(h_prev[:,None,:])
            c.append(c_prev[:,None,:])    
        
        return torch.cat(h, dim = 1), torch.cat(c, dim = 1)