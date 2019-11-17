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

class Embedding(nn.Module):
    def __init__(self, options, weights = None):
        super(Embedding, self).__init__()
        
        
        # Define the Embedding Layer
        self.n_type = options['n_type']
        self.n_dim = options['n_dim']
        self.fixed = options['fixed'] 
        
        # Using a pre-trained Embedding Matrix
        if weights is not None:
            self.layer = nn.Embedding.from_pretrained(weights, freeze = self.fixed)
        else:
            self.layer = nn.Embedding(self.n_type, self.n_dim)
            nn.init.normal_(self.layer.weight, std = 0.5)
    
    def forward(self, input):
        return self.layer(input)
    