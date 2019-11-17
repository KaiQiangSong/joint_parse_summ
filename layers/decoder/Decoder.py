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
from .Decoder_Flat import Decoder_Flat
from .Decoder_FlatLVT import Decoder_FlatLVT
from .Decoder_FlatParse import Decoder_FlatParse
from .Decoder_deRNNG import Decoder_deRNNG

class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        self.layer = eval(options['type'])(options)
    
    def forward(self, *argv, **kwargs):
        return self.layer(*argv, **kwargs)