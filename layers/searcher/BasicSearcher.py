import itertools

from layers.utils import *
from layers.data_process import OpMapping_De

class BasicSearcher(object):
    def __init__(self, options):
        self.method = options['test']['method']
        self.beam_size = options['test']['beam_size']
        self.answer_size = options['test']['answer_size']
        self.biGramTrick = options['test']['biGramTrick']
        self.gamma = options['test']['gamma']
        self.maxLength = options['test']['maxLength']
        self.type = None
        self.BOS = None
        
        self.Embs = None
        self.generator = None
        self.LVT = None
        self.conditions = {}
        if "Conditions" in options['test']:
            self.conditions_Set(options['test']['Conditions'])
        
    def Emb_Set(self, Emb):
        self.Emb = Emb
        
    def setType(self, fType):
        self.type = fType
        self.BOS = fType.BOS
    
    def Generator_Set(self, generator):
        self.generator = generator
        
    def conditions_Set(self, conditions):
        self.conditions = conditions
    
    def conditions_Reset(self):
        self.conditions = {}
        
    def condition_new(self, key, value):
        self.conditions[key] = value
        return value
    
    def conditions_Get(self, key):
        if key in self.conditions:
            return self.conditions[key]
        return None
    
    def LVT_Set(self, LVT):
        self.LVT = LVT
        
    def state_init(self):
        return self.type.state_init()
    
    def state_pack(self, s):
        return self.type.state_pack(s)
    
    def state_process(self, s_old, s_new, action):
        return self.type.state_process(s_old, s_new, action)
    
    def state_unpack(self, s):
        return self.type.state_unpack(s)
    
    def cond(self, o, s, c = None):
        return self.type.cond(o, s, c)

    def getPred(self, x):
        return self.type.getPred(x)
    
    def getAtt(self, x):
        return self.type.getAtt(x)
    
    def lastToken(self, seq, conditions):
        return self.type.lastToken(seq, conditions)
    
    def checkEnd(self, Sequence, State_Pass):
        return self.type.checkEnd(Sequence, State_Pass)
    
    @staticmethod
    def atom(l):
        res = []
        for e in l:
            res.append(torch2numpy(e).flatten() if e is not None else None)
        return res
    
    def do_bigramTrick(self, preds, ids, beam_size, token, bi_in, bi_old, gamma, pool):
        if bi_in is None:
            bi_in = set()
        
        ratio = gamma / (len(bi_in) + 1e-8)
        ids_update = topKIndexes(preds[ids], beam_size, ratio)
        ids = ids[ids_update]
        
        bi_new = list(itertools.product([token], pool[ids].tolist()))
        trick_item = [int((biGram in bi_in) &(biGram not in bi_old)) for biGram in bi_new]
        trick_item = np.asarray(trick_item, dtype = np.float32)
        
        dist = preds[ids]
        dist += ratio * trick_item
        
        ids_update = topKIndexes(dist, beam_size)
        ids = ids[ids_update]
        
        #bi_next = [copy.deepcopy(bi_old).union(biGram) if ((biGram in bi_in) & (biGram not in bi_old)) else copy.deepcopy(bi_old) for biGram in bi_new]
        bi_next = []
        for i in range(len(bi_new)):
            if i in ids_update.tolist():
                biGram = bi_new[i]
                bi_next.append(copy.deepcopy(bi_old).union(biGram) if ((biGram in bi_in) & (biGram not in bi_old)) else copy.deepcopy(bi_old))
        
        dist = dist[ids_update]
        return dist, ids, bi_next
    
    def no_bigramTrick(self, preds, ids, beam_size):
        ids_update = topKIndexes(preds[ids], beam_size)
        ids = ids[ids_update]
        return preds[ids], ids
    
    def getInputs(self, Token, Lengths):
        return self.type.getInputs(Token, Lengths, self.Emb)

    def search(self):
        pass
          