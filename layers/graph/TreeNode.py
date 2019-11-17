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
from .TreeNode_De import TreeNode_De

class TreeNode(nn.Module):
    def __init__(self, options):
        super(TreeNode, self).__init__()
        self.type = options['type']
        self.layer = eval(self.type)(options)
        
    def forward(self, *argv, **kwagrs):
        return self.layer(*argv, **kwagrs)
        