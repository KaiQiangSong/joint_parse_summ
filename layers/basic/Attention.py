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


class Attention(nn.Module):
    EPS = 1e-8
    def __init__(self, options, weights = None):
        super(Attention, self).__init__()
        self.n_encoder = options['n_encoder']
        self.n_decoder = options['n_decoder']
        self.n_att = options['n_att']
        self.layer_en = nn.Linear(self.n_encoder, self.n_att)
        self.layer_de = nn.Linear(self.n_decoder, self.n_att, bias = False)
        self.layer_att = nn.Linear(self.n_att, 1)
        if weights is not None:
            self.layer_en.weights = nn.Parameter(weights[0])
            self.layer_de.weights = nn.Parameter(weights[1])
            self.layer_att.weights = nn.parameter(weights[2])
        
        self.fixed = options['fixed']
        if self.fixed:
            self.layer_en.weight.requires_grad = False
            self.layer_de.weight.requires_grad = False
            self.layer_att.weight.requires_grad = False
        
    def forward(self, h_e, h_d, mask):
        '''
            h_e is BxLx*
            h_d is BxLx*
        '''
        # h_d : BxTx*xD
        # h_e : Bx*xSxD
        
        h_d = self.layer_de(h_d)[:,:,:,None].permute(0,1,3,2)
        h_e = self.layer_en(h_e)[:,:,:,None].permute(0,3,1,2)
        
        # activation BxTxSxD
        activation = F.tanh(h_d + h_e)
        activation = self.layer_att(activation)
        shp = activation.size()
        
        # activation BxTxS
        # Doing Softmax with variant of length
        activation = activation.view(shp[0],shp[1],-1)
        result = F.softmax(activation, dim = 2) * mask[:,None,:]
        result = result / (result.sum(dim = 2, keepdim=True) + self.EPS)
        return result