#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch.nn as nn
import torch.nn.functional as F

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.basic import Linear

class TreeNode_De(nn.Module):
    def __init__(self, options):
        super(TreeNode_De, self).__init__()
        self.layer = Linear(options)
    
    def forward(self, data):
        return F.relu(self.layer(data))