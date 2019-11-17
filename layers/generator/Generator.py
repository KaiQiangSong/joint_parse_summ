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
from .Generator_Flat import Generator_Flat
from .Generator_FlatParse import Generator_FlatParse
from .Generator_deTree import Generator_deTree
          
class Generator(nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.layer = eval(options['type'])(options)
        
    def forward(self, *argv, **kwargs):
        return self.layer(*argv, **kwargs)