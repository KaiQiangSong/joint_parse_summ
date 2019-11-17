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
from .AStar import AStar
from .BFS_BEAM import BFS_BEAM
from .Debug import Debug
from .Prob import Prob
from .BasicSearcher import BasicSearcher

class Searcher(object):
    def __init__(self, options):
        self.searcher = eval(options['test']['method'])(options)
    
    def search(self, *argv, **kwargs):
        return self.searcher.search(*argv, **kwargs)
    
    def setType(self,*argv, **kwargs):
        return self.searcher.setType(*argv, **kwargs)
    
    def Emb_Set(self, *argv, **kwargs):
        return self.searcher.Emb_Set(*argv, **kwargs)
    
    def Generator_Set(self, *argv, **kwargs):
        return self.searcher.Generator_Set(*argv, **kwargs)
    
    def LVT_Set(self, *argv, **kwargs):
        return self.searcher.LVT_Set(*argv, **kwargs)