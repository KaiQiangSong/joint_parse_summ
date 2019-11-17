#==============================#
#    System Import             #
#==============================#


#==============================#
#    Platform Import           #
#==============================#
import torch
from torch.autograd import Variable

#==============================#
#    Class/Layer Part Import   #
#==============================#
from .BasicSearcher import BasicSearcher



class Debug(BasicSearcher):
    def __init__(self, options):
        super(Debug, self).__init__(options)
    
    def search(self, state_below, action, len_action):
        Lengths = Variable(torch.LongTensor([1]))
        if torch.cuda.is_available():
            Lengths = Lengths.cuda()
        
        outputs = []
        States = self.state_init()
        for i in range(len_action):
            Token = action[i].view(1,1)
            Inputs = self.getInputs(Token, Lengths)
            output, States = self.generator(self.LVT, state_below, Inputs, States)
            outputs.append(output)
        
        m = len(outputs[0])
        results = []
        for i in range(m):
            result = []
            for j in range(len_action):
                result.append(outputs[j][i])
            result = torch.cat(result, dim = 1)
            results.append(result)
        return results
            