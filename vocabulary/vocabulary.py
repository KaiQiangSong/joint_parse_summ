import numpy as np

from argparse import Namespace


from utility.utility import *
from mylog.mylog import mylog
from collections import OrderedDict
from options_process.optionsLoader import optionsLoader

class Vocabulary(object):
    def __init__(self, options):
        self.cased = options['cased']
        self.rmDigit = options['rmDigit']
        self.sortBy = options['sortBy']
        self.minFreq = options['minFreq']
        self.dim = options['dim']
        self.init_path = options['initPath']
        self.inputCorpus = options['inputCorpus']
        self.outputCorpus = options['outputCorpus']
        
        self.typeFreq = self.setup()
        number, n_in, n_out, w2i, i2w, i2e = self.initialize(self.typeFreq)
        self.full_size = number
        self.n_in = n_in
        self.n_out = n_out
        self.w2i = w2i
        self.i2w = i2w
        self.i2e = i2e
        
    def accumulate(self, fileName, Counter):
        f = open(fileName, 'r', encoding = 'utf-8')
        for line in f:
            tokens = line.split()
            for token in tokens:
                word = token
                if not self.cased:
                    word = word.lower()
                if self.rmDigit:
                    word = remove_digits(word)
                if word not in Counter:
                    Counter[word] = 1
                else:
                    Counter[word] += 1
        f.close()
        return Counter
    
    def setup(self):
        typeFreq_input = {}
        typeFreq_output = {}
        if (type(self.inputCorpus) == str):
            typeFreq_input = self.accumulate(self.inputCorpus, typeFreq_input)
        else:
            for fileName in self.inputCorpus:
                typeFreq_input = self.accumulate(fileName, typeFreq_input)
        if (type(self.outputCorpus) == str):
            typeFreq_output = self.accumulate(self.outputCorpus, typeFreq_output)
        else:
            for fileName in self.outputCorpus:
                typeFreq_output = self.accumulate(fileName, typeFreq_output)
        
        typeFreq_Full = {}
        '''
        for key, value in typeFreq_input.items():
            if (key in typeFreq_output) and (typeFreq_input[key] + typeFreq_output[key] >= minFreq):
                typeFreq_Full[key] = (typeFreq_input[key], typeFreq_output[key], typeFreq_input[key] + typeFreq_output[key])
            elif typeFreq_input[key] >= minFreq :
                typeFreq_Full[key] = (typeFreq_input[key], 0, typeFreq_input[key])
        
        for key, value in typeFreq_output.items():
            if (key not in typeFreq_input) and (typeFreq_output[key] >= minFreq):
                typeFreq_Full[key] = (0, typeFreq_output[key], typeFreq_output[key])
        '''
        for key, value in typeFreq_output.items():
            if (typeFreq_output[key] >= self.minFreq):
                if (key in typeFreq_input):
                    typeFreq_Full[key] = (typeFreq_input[key], typeFreq_output[key], typeFreq_input[key] + typeFreq_output[key])
                else:
                    typeFreq_Full[key] = (0, typeFreq_output[key], typeFreq_output[key])
        
        for key, value in typeFreq_input.items():
            if (key not in typeFreq_output) and (typeFreq_input[key] >= self.minFreq * 5):
                typeFreq_Full[key] = (typeFreq_input[key], 0, typeFreq_input[key])
        
        if self.sortBy == 'input':
            select = 0
            another = 1
        elif self.sortBy == 'output':
            select = 1
            another = 0
        else:
            select = 2
            another = 1
            
        typeFreq = OrderedDict(sorted(typeFreq_Full.items(), key = lambda x:(-x[1][select], -x[1][another])))
        return typeFreq
    
    def loadEmbedding(self):
        embedding = {}
        f = open(self.init_path, 'r', encoding = 'utf-8')
        for line in f:
            items = line.split()
            word = items[0]
            if not self.cased:
                word = word.lower()
            if self.rmDigit:
                word = remove_digits(word)
            emb = list(map(lambda x:float(x), items[1:]))
            emb = np.array(emb, dtype = np.float32)
            embedding[word] = emb        
        return embedding
    
    def initialize(self, typeFreq):
        if (self.init_path != None) and (self.init_path != ''):
            pretrained = self.loadEmbedding()
        else:
            pretrained = {}
            
        i2w = ['<unk>','<bos>','<eos>']
        w2i = {
            '<unk>':0,
            '<bos>':1,
            '<eos>':2,
            }
        number = 3
        n_in = 0
        n_out = 0
        for key, value in typeFreq.items():
            if key not in w2i:
                i2w.append(key)
                w2i[key] = number
                number += 1
            if (key != '<unk>') and (key != '<bos>') and (key != '<eos>'):
                if (value[0] > 0):
                    n_in += 1
                if (value[1] > 0):
                    n_out += 1
        i2e = np.empty((0, self.dim), dtype = np.float32)
        for Index in range(0, number):
            type = i2w[Index]
            if (type in pretrained):
                embedding = pretrained[type].reshape(1, self.dim)
            else:
                embedding = random_weights(1, self.dim, 0.5)
            i2e = np.append(i2e, embedding, axis = 0)
        
        return number, n_in + 3, n_out + 3, w2i, i2w, i2e

