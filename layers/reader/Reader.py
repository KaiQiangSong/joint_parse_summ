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
from .Reader_LSTM2 import Reader_LSTM2

class Reader(nn.Module):
    def __init__(self, options):
        super(Reader, self).__init__()
        self.layer = eval(options['type'])(options)
        
    def forward(self, *argv, **kwargs):
        return self.layer(*argv, **kwargs)