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
from .LSTM2_MeanDiff_FlatToken import *
from .LSTM2_MeanDiff_FlatParse import *
from .LSTM2_MeanDiff_deRNNG import *

class framework(nn.Module):
    '''
        I am just a trigger 
    '''
    def __init__(self, options, log, emb_tok_init = None):
        super(framework, self).__init__()
        self.layer = eval(options['network']['type'])(options, log, emb_tok_init)
        
    def setLVT(self, LVT):
        return self.layer.setLVT(LVT)
        
    def forward(self, *argv, **kwargs):
        return self.layer(*argv, **kwargs)
    
    def getLoss(self, *argv, **kwargs):
        return self.layer.getLoss(*argv, **kwargs)
    
    def genSummary(self, *argv, **kwargs):
        return self.layer.genSummary(*argv, **kwargs)
    
    def getProb(self, *argv, **kwargs):
        return self.layer.getProb(*argv, **kwargs)
    
    def debug(self, *argv, **kwargs):
        return self.layer.debug(*argv, **kwargs)