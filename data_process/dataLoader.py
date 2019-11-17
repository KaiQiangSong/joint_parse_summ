import codecs, random

from mylog.mylog import mylog
from utility.utility import *
from .utils import *

OTHER, REDUCE_L, REDUCE_R, GEN = 0, 1, 2, 3

class dataSet(object):
    BATCH_SIZE_DEFAULT = 64
    #BATCH_SIZE_DEFAULT = 32
    #BATCH_SIZE_DEFAULT = 16
     
    def __init__(self, name, options, log, Vocab):
        self.name = name
        self.method = options['method']
        self.srcFeatList = options['srcFeatList']
        self.refFeatList = options['refFeatList']
        
        self.log = log 
        self.Vocab = Vocab
        
        self.path = options['Parts'][name]['path']
        self.sorted = options['Parts'][name]['sorted']
        self.shuffled = options['Parts'][name]['shuffled']
        
        self.data_applied = False
        self.batch_applied = False
        self.Data = {}
        self.Batch = {"batch_size":self.BATCH_SIZE_DEFAULT}
        
        
    def setSrcFeat(self, value):
        self.srcFeatList = value
    def setRefFeat(self, value):
        self.refFeatList = value
    def setMethod(self, method):
        self.method = method
    def setBatchSize(self, batch_size):
        self.Batch['batch_size'] = batch_size
        
    def load(self):
        srcFile = codecs.open(self.path+'.Ndocument','r', encoding='utf-8')
        refFile = codecs.open(self.path+'.Nsummary','r', encoding='utf-8')
        ErrorFile = codecs.open(self.path+'.ErrorList','w', encoding='utf-8')
        EidFile = codecs.open(self.path+'.Eid','w', encoding='utf-8')
        data, anno = [], []
        
        srcFeat = len(self.srcFeatList) > 0
        refFeat = len(self.refFeatList) > 0
        if (srcFeat):
            srcFeatFile = codecs.open(self.path+'.dfeature','r',encoding='utf-8')
            sfeat = []
        else:
            sfeat = None
        if 'test' not in self.name and (refFeat):
            refFeatFile = codecs.open(self.path+'.sfeature','r',encoding='utf-8')
            rfeat = []
        else:
            rfeat = None
        
        Index, total = 0, 0
        while True:
            Index += 1
            #print Index
            
            # Readlines
            srcLine = srcFile.readline()
            if (not srcLine):
                break
            refLine = refFile.readline()
            if (not refLine):
                break
            if srcFeat:
                srcFeatLine = srcFeatFile.readline()
                if (not srcFeatLine):
                    break
            if 'test' not in self.name and (refFeat):
                refFeatLine = refFeatFile.readline()
                if (not refFeatLine):
                    break
            
            # Processing Input Sequences
            srcLine = remove_digits(srcLine.strip()).lower()
            refLine = remove_digits(refLine.strip()).lower()
            
            if 'train' in self.name or 'val' in self.name:
                if (len(srcLine.split()) >= 100) or (len(refLine.split()) >= 50):
                    #print("too long")
                    continue
            
            if 'train' in self.name:
                match = len(set(srcLine.split()) & set(refLine.split()))
                if match < 3:
                    #print("No match")
                    continue
            
            if (len(srcLine.split()) < 1) or (len(refLine.split()) < 1):
                #print('Short')
                continue
            
            document = Sentence2ListOfIndex(srcLine, self.Vocab)
            summary = Sentence2ListOfIndex(refLine, self.Vocab)
            
            '''
            if (len(summary) > 300):
                #print 'Long'
                continue
            '''
            
            if (srcFeat):
                src_feat = eval(srcFeatLine.strip())
                # Check Length Match in source
                length = len(document)
                lengths = [len[fet] for fet in src_feat.values()]
                if (len(Set(lengths + [length])) > 1):
                    total += 1
                    #print("Source Length Doesn't match")
                    print("Source Length Doesn't match", total, Index, length, lengths, file = ErrorFile)
                    print(srcLine, file = ErrorFile)
                    print(Index-1, file = ErrorFile)
                    continue
            
            if 'test' not in self.name and (refFeat):
                ref_feat = eval(refFeatLine.strip())
                # Check Length Match in Reference
                length = len(summary)
                length_gen = sum([1 if it[0] == GEN else 0 for it in ref_feat['action_map']])
                if (length != length_gen):
                    total += 1
                    tokens = [it[1] for it in ref_feat['action_map'] if it[0] == GEN]
                    #print("Reference Length Doesn't match", total, Index, length, length_gen)
                    print("Reference Length Doesn't match", total, Index, length, length_gen, file=ErrorFile)
                    print(refLine, file=ErrorFile)
                    print(tokens, file=ErrorFile)
                    print(Index-1, file=ErrorFile)
                    continue
            
            data.append(document)
            anno.append(summary)
            if (srcFeat):
                sfeat.append(src_feat)
            if 'test' not in self.name and (refFeat):
                rfeat.append(ref_feat)
        return len(anno), (data, anno, sfeat, rfeat)
    
    def sortByLength(self):
        self.log.log('Start sorting by length')
        data = self.Data['data'][0]
        anno = self.Data['data'][1]
        srcFeat = len(self.srcFeatList) > 0
        refFeat = len(self.refFeatList) > 0
        if (srcFeat):
            sfeat = self.Data['data'][2]
        if (refFeat):
            rfeat = self.Data['data'][3]
        
        number = len(anno)
        
        lengths =  [(len(anno[Index]), Index) for Index in range(number)]
        sorted_lengths = sorted(lengths)
        sorted_Index = [d[1] for d in sorted_lengths]
        
        data_new = [data[sorted_Index[Index]] for Index in range(number)]
        anno_new = [anno[sorted_Index[Index]] for Index in range(number)]
        if (srcFeat):
            sfeat_new = [sfeat[sorted_Index[Index]] for Index in range(number)]
        else:
            sfeat_new = None
        
        if (refFeat):
            rfeat_new = [rfeat[sorted_Index[Index]] for Index in range(number)]
        else:
            rfeat_new = None
            
        self.Data['data'] = (data_new, anno_new, sfeat_new, rfeat_new)
        self.log.log('Stop sorting by length')
    
    def shuffle(self):
        self.log.log('Start Shuffling')
        
        data = self.Data['data'][0]
        anno = self.Data['data'][1]
        srcFeat = len(self.srcFeatList) > 0
        refFeat = len(self.refFeatList) > 0
        if (srcFeat):
            sfeat = self.Data['data'][2]
        if (refFeat):
            rfeat = self.Data['data'][3]
        
        number = len(anno)
        
        shuffle_Index = list(range(number))
        random.shuffle(shuffle_Index)
        
        data_new = [data[shuffle_Index[Index]] for Index in range(number)]
        anno_new = [anno[shuffle_Index[Index]] for Index in range(number)]
        if (srcFeat):
            sfeat_new = [sfeat[shuffle_Index[Index]] for Index in range(number)]
        else:
            sfeat_new = None
        
        if (refFeat):
            rfeat_new = [rfeat[shuffle_Index[Index]] for Index in range(number)]
        else:
            rfeat_new = None
        
        self.Data['data'] = (data_new, anno_new, sfeat_new, rfeat_new)
        self.log.log('Finish Shuffling')
        
    def genBatches(self):
        batch_size = self.Batch['batch_size']
        data = self.Data['data'][0]
        anno = self.Data['data'][1]
        srcFeat = len(self.srcFeatList) > 0
        refFeat = len(self.refFeatList) > 0
        
        if (srcFeat):
            sfeat = self.Data['data'][2]
        if (refFeat):
            rfeat = self.Data['data'][3]
        
        number = self.number()
        number_batch = number // batch_size
        batches = []
        for bid in range(number_batch):
            data_i = data[bid * batch_size: (bid + 1) * batch_size]
            anno_i = anno[bid * batch_size: (bid + 1) * batch_size]
            if (srcFeat):
                sfeat_i = sfeat[bid * batch_size: (bid + 1) * batch_size]
            else:
                sfeat_i = None
            if (refFeat):
                rfeat_i = rfeat[bid * batch_size: (bid + 1) * batch_size]
            else:
                rfeat_i = None
            batches.append((data_i, anno_i, sfeat_i, rfeat_i))
        if (number_batch * batch_size < number):
            data_i = data[number_batch * batch_size:]
            anno_i = anno[number_batch * batch_size:]
            if (srcFeat):
                sfeat_i = sfeat[number_batch * batch_size:]
            else:
                sfeat_i = None
            if (refFeat):
                rfeat_i = rfeat[number_batch * batch_size:]
            else:
                rfeat_i = None
            batches.append((data_i, anno_i, sfeat_i, rfeat_i))
            number_batch += 1
        
        self.Batch['n_batches'] = number_batch
        self.Batch['batches'] = batches
        self.Batch['index'] = list(range(number_batch))
            
        
    def apply_data(self):
        self.log.log('Applying Data')
        if self.method == 'build':
            self.log.log('Building dataset %s from orignial text documents'%(self.name))
            number, data = self.load()
            self.Data['number'] = number
            self.Data['data'] = data
            saveToPKL(self.path + '.data', self.Data)
            self.data_applied = True
            self.log.log('Finish Loading dataset %s'%(self.name))
        elif self.method == 'load':
            self.log.log('Loading Subset %s from PKL File'%(self.name))
            self.Data = loadFromPKL(self.path+'.data')
            self.data_applied = True
            self.log.log('Finish Loading Subset %s'%(self.name))
        return 
    
    def apply_batch(self):
        self.log.log('Applying Batches')
        if self.method == 'build':
            if not self.data_applied:
                self.apply_data()
                 
            if (self.sorted):
                self.sortByLength()
                
            if (self.shuffled):
                self.shuffle()
            
            self.log.log('Generating Batches')    
            self.genBatches()
            saveToPKL(self.path+'.batches', self.Batch)
            self.batch_applied = True
            self.log.log('Finish Gnerating Batches')
        elif self.method == 'load':
            self.log.log('Loading Batches of %s from PKL File'%(self.name))
            self.Batch = loadFromPKL(self.path+'.batches')
            self.batch_applied = True
            self.log.log('Finish Loading Batches of %s'%(self.name))
        return
    
    def number(self):
        if self.data_applied:
            return self.Data['number']
        if self.batch_applied:
            return self.Batch['batch_size'] * (self.Batch['n_batch'] - 1) + len(self.Batch['batches'][-1][1])
        self.apply_data()
        return self.Data['number']
    
    def data(self):
        if not self.data_applied:
            self.apply_data()
        return self.Data['data']
    
    def n_batches(self):
        if not self.batch_applied:
            self.apply_batch()
        return self.Batch['n_batches']
    
    def batches(self):
        if not self.batch_applied:
            self.apply_batch()
        return self.Batch['batches']
    
    def batchShuffle(self):
        if not self.batch_applied:
            self.apply_batch()
        random.shuffle(self.Batch['index'])
    
    def get_Kth_Batch(self, K):
        if not self.batch_applied:
            self.apply_batch()
        return self.Batch['batches'][self.Batch['index'][K]]
    
    def get_first_K_Batches(self, K):
        if not self.batch_applied:
            self.apply_batch()
        K = min(K, self.Batch['n_batches'])
        data, anno = [], []
        srcFeat = len(self.srcFeatList) > 0
        refFeat = len(self.refFeatList) > 0
        
        if srcFeat:
            sfeat = []
        else:
            sfeat = None
        
        if 'test' not in self.name and (refFeat):
            rfeat = []
        else:
            rfeat = None
            
        for i in range(K):
            data_i, anno_i, sfeat_i, rfeat_i = self.get_Kth_Batch(i)
            data.append(data_i)
            anno.append(anno_i)
            if srcFeat:
                sfeat.append(sfeat_i)
            if 'test' not in self.name and (refFeat):
                rfeat.append(rfeat_i)
        return (data, anno, sfeat, rfeat)
    
    def get_Kth_instance(self, K):
        if not self.data_applied:
            self.apply_data()
            
        data = self.Data['data'][0][K]
        anno = self.Data['data'][1][K]
            
        srcFeat = len(self.srcFeatList) > 0
        refFeat = len(self.refFeatList) > 0
        
        if srcFeat:
            sfeat = self.Data['data'][2][K]
        else:
            sfeat = None
        
        if 'test' not in self.name and (refFeat):
            rfeat = self.Data['data'][3][K]
        else:
            rfeat = None
        return (data, anno, sfeat, rfeat)
    
    def get_first_K_instances(self, K, Shuffle = False):
        if not self.data_applied:
            self.apply_data()
        
        data = self.Data['data']
        n_data = self.Data['number']
        randomIndex = range(n_data)
        if Shuffle:
            random.shuffle(randomIndex)
        
        K = min(K, n_data)
        
        srcFeat = len(self.srcFeatList) > 0
        refFeat = len(self.refFeatList) > 0
        data = []
        anno = []
        if srcFeat:
            sfeat = []
        else:
            sfeat = None
        
        if 'test' not in self.name and (refFeat):
            rfeat = []
        else:
            rfeat = None
        
        for Index in range(K):
            data_i, anno_i, sfeat_i, rfeat_i = self.get_Kth_instance(randomIndex[Index])
            data.append(data_i)
            anno.append(anno_i)
            if srcFeat:
                sfeat.append(sfeat_i)
            if 'test' not in self.name and (refFeat):
                rfeat.append(rfeat_i)
        return (data, anno, sfeat, rfeat)


class dataLoader(object):
    def __init__(self, log, options, Vocab):
        self.name = options['name']
        self.log = log
        self.Parts = {}
        for name, partConfig in options['Parts'].items():
            self.Parts[name] = dataSet(name, options, log, Vocab)
        
    def batchShuffle(self, part = 'train'):
        self.Parts[part].batchShuffle()
        
    def get_Kth_Batch(self, K, part = 'train'):
        return self.Parts[part].get_Kth_Batch(K)
    
    def get_Kth_instance(self, K, part = 'train'):
        return self.Parts[part].get_Kth_instance(K)
    
    def get_first_K_instances(self, K, part = 'train'):
        return self.Parts[part].get_first_K_instances(K)
        
    def get_random_K_instances(self, K, part = 'train'):
        return self.Parts[part].get_first_K_instances(K, Shuffle = True)
    