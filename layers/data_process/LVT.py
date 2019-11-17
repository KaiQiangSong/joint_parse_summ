#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#

#==============================#
#    Class/Layer Part Import   #
#==============================#
from .Padding import *

class LargeVocabularyTrick(object):
    '''
        ****Important Information****
        The Sequences for Initialization should be in OPI(Operation Index) style
        ****Important Information****
    '''
    def initialize(self, Pool, seqs, options):
        for seq in seqs:
            Pool = Pool.union(set(seq))
            
        self.Pool = sorted(list(Pool))
        self.size = len(self.Pool)
        
        Dict = {}
        Index = 0
        for token in self.Pool:
            Dict[token] = Index
            Index += 1
        
        self.Dict = Dict
        self.Size = Index

        pointer = []
        for seq in seqs:
            pointer.append([self.Dict[token] for token in seq])
        
        pointer, _ = Padding.padding(pointer)
        self.Pointer = pointer

        
    def __init__(self, seqs, options):
        self.UNK = 0
        Pool = set(range(options['network']['n_vocab']))
        self.initialize(Pool, seqs, options)        
                
    def sharp(self, seqs):
        sharped_seq = [[self.Dict[act] if act in self.Dict else self.UNK for act in seq] for seq in seqs]
        return sharped_seq
        
class ActionPool_De(LargeVocabularyTrick):
    '''
        Operations Included:
            0. OTHER
            1. REDUCE_L
            2. REDUCE_R
            3~5002. GEN + X
            5003~? GEN + Y
        Data Included:
            Pool: A CandidatedPool for this batch
            Dict: A Dictionary return index for the value
            Pointer: A Pointer for where the source token is in the Pool
    '''
    def __init__(self, seqs, options):
        self.UNK = 3
        Pool = set(range(3 + options['network']['n_vocab']))
        self.initialize(Pool, seqs, options)