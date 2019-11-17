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
from layers.basic.LSTM import LSTM

class Encoder(nn.Module):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.layer = eval(options['type'])(options)
    
    def forward(self, *argv, **kwargs):
        return self.layer(*argv, **kwargs)