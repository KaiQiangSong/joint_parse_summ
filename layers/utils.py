import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.autograd as autograd
from torch.autograd import Variable
from utility.utility import *
from cmath import sqrt


def PadSequence(BoS):
    # Get the Lengths and Ids
    lengths, IDs = zip(*sorted(zip([len(s) for s in BoS], range(len(BoS))), key=lambda x:-x[0]))
    _, bIDs = zip(*sorted(zip(IDs, range(len(IDs))), key=lambda x:x[0]))
    Padded = torch.stack([torch.cat([Variable(torch.LongTensor(BoS[IDs[i]]), requires_grad = False),
                                     Variable(torch.LongTensor([0] * (lengths[0] - lengths[i])), requires_grad = False)])
                                     for i in range(len(lengths))])
    lengths = [lengths[it] for it in bIDs]
    IDs = Variable(torch.LongTensor(IDs), requires_grad = False)
    bIDs = Variable(torch.LongTensor(bIDs), requires_grad = False)
    if torch.cuda.is_available():
        Padded = Padded.cuda()
        IDs = IDs.cuda()
        bIDs = bIDs.cuda()
    return Padded, lengths, IDs, bIDs

def Pad2DMatrix(BoM):
    #print BoM
    dim = BoM[0].size()[1]
    lengths, IDs = zip(*sorted(zip([m.size()[0] for m in BoM], range(len(BoM))), key= lambda x:-x[0]))
    _, bIDs = zip(*sorted(zip(IDs, range(len(IDs))), key = lambda x:x[0]))
    if torch.cuda.is_available():
        Padded = torch.stack([torch.cat([BoM[IDs[i]], Variable(torch.zeros((lengths[0] - lengths[i], dim)), requires_grad = False).cuda()], dim = 0) if lengths[i] < lengths[0] else BoM[IDs[i]] for i in range(len(lengths))])
    else:
        Padded = torch.stack([torch.cat([BoM[IDs[i]], Variable(torch.zeros((lengths[0] - lengths[i], dim)), requires_grad = False)], dim = 0) if lengths[i] < lengths[0] else BoM[IDs[i]] for i in range(len(lengths))])
    lengths = [lengths[it] for it in bIDs]
    IDs = Variable(torch.LongTensor(IDs), requires_grad = False)
    bIDs = Variable(torch.LongTensor(bIDs), requires_grad = False)
    if torch.cuda.is_available():
        IDs = IDs.cuda()
        bIDs = bIDs.cuda() 
    return Padded, lengths, IDs, bIDs

'''
def genMask(lengths):
    maxLength = max(lengths)
    if torch.cuda.is_available():
        seq = [torch.cat([Variable(torch.ones(it), requires_grad = False).cuda(), Variable(torch.zeros(maxLength - it), requires_grad = False).cuda()]) for it in lengths]
    else:
        seq = [torch.cat([Variable(torch.ones(it), requires_grad = False), Variable(torch.zeros(maxLength - it), requires_grad = False)]) for it in lengths]
    mask = torch.stack(seq)
    return mask
'''

def genMask(lengths):
    maxLength = int(torch.max(lengths))
    lengths = [int(it) for it in lengths]
    mask = torch.stack([torch.ones(maxLength) if maxLength == lengths[i] else torch.cat([torch.ones(lengths[i]), torch.zeros(maxLength - lengths[i])]) for i in range(len(lengths))])
    mask = Variable(mask, requires_grad = False)
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask

def PadOperations(BoO):
    lengths, IDs = zip(*sorted(zip([len(s) for s in BoO], range(len(BoO))), key=lambda x:-x[0]))
    _, bIDs = zip(*sorted(zip(IDs, range(len(IDs))), key=lambda x:x[0]))
    BoO = [zip(*instance) for instance in BoO]
    if torch.cuda.is_available():
        Padded = (torch.stack([torch.cat([Variable(torch.LongTensor(BoO[IDs[i]][0]), requires_grad = False).cuda(), Variable(torch.LongTensor([0] * (lengths[0] - lengths[i])), requires_grad = False).cuda()]) for i in range(len(lengths))]),
                  torch.stack([torch.cat([Variable(torch.LongTensor(BoO[IDs[i]][1]), requires_grad = False).cuda(), Variable(torch.LongTensor([0] * (lengths[0] - lengths[i])), requires_grad = False).cuda()]) for i in range(len(lengths))]))
    else:
        Padded = (torch.stack([torch.cat([Variable(torch.LongTensor(BoO[IDs[i]][0]), requires_grad = False), Variable(torch.LongTensor([0] * (lengths[0] - lengths[i])), requires_grad = False)]) for i in range(len(lengths))]),
                  torch.stack([torch.cat([Variable(torch.LongTensor(BoO[IDs[i]][1]), requires_grad = False), Variable(torch.LongTensor([0] * (lengths[0] - lengths[i])), requires_grad = False)]) for i in range(len(lengths))]))
    lengths = [lengths[it] for it in bIDs]
    IDs = Variable(torch.LongTensor(IDs), requires_grad = False)
    bIDs = Variable(torch.LongTensor(bIDs), requires_grad = False)
    
    if torch.cuda.is_available():
        IDs = IDs.cuda()
        bIDs = bIDs.cuda()
        
    return Padded, lengths, IDs, bIDs 

def oneHot(input, class_number):
    shp = input.size()
    input = input.view(-1, 1)
    batch_size = input.size()[0]
    
    result = Variable(torch.zeros(batch_size, class_number), requires_grad = False)
    if torch.cuda.is_available():
        result = result.cuda()
    result = result.scatter_(1, input, 1)
    
    return result.view(shp+(class_number,))

def PaddingSequenceVectors(BoS):
    dim = BoS[0].size()[1]
    lengths = [S.size()[0] for S in BoS]
    maxLength = max(lengths)
    #newBoS = torch.stack([torch.cat([BoS[i], Variable(torch.zeros((maxLength - lengths[i], dim)))]) if lengths[i] < maxLength else BoS[i] for i in range(len(BoS))])
    newBoS = []
    for i in range(len(BoS)):
        if lengths[i] < maxLength:
            if torch.cuda.is_available():
                newBoS.append(torch.cat([BoS[i], Variable(torch.zeros((maxLength-lengths[i], dim)), requires_grad = False).cuda()]))
            else:
                newBoS.append(torch.cat([BoS[i], Variable(torch.zeros((maxLength-lengths[i], dim)), requires_grad = False)]))
        else:
            newBoS.append(BoS[i])
    newBoS = torch.stack(newBoS)
    return newBoS

def PaddingIndex(BoS):
    lengths = [len(s) for s in BoS]
    maxLength = max(lengths)
    newBoS = []
    for i in range(len(BoS)):
        if lengths[i] < maxLength:
            if torch.cuda.is_available():
                newBoS.append(torch.cat([Variable(torch.LongTensor(BoS[i]), requires_grad = False).cuda(), Variable(torch.LongTensor([0] * (maxLength-lengths[i])), requires_grad = False).cuda()]))
            else:
                newBoS.append(torch.cat([Variable(torch.LongTensor(BoS[i]), requires_grad = False), Variable(torch.LongTensor([0] * (maxLength-lengths[i])), requires_grad = False)]))
        else:
            if torch.cuda.is_available():
                newBoS.append(Variable(torch.LongTensor(BoS[i]), requires_grad = False).cuda())
            else:
                newBoS.append(Variable(torch.LongTensor(BoS[i])), requires_grad = False)
    newBoS = torch.stack(newBoS)
    return newBoS

def Indexing(operations, h):
    dim = h.size()[2]
    ZERO = Variable(torch.zeros(dim), requires_grad = False)
    if torch.cuda.is_available():
        ZERO = ZERO.cuda()
            
    bidx = 0
    newH = []
    for actions in operations:
        h_i = []
        pt = -1
        for oper, id in actions:
            if (oper == 2):
                pt += 1
            if (pt < 0):
                h_i.append(ZERO)
            else:
                h_i.append(h[bidx,pt,:])
        h_i = torch.stack(h_i)
        #print h_i.size()
        newH.append(h_i)
        bidx += 1
    return PaddingSequenceVectors(newH)

def PaddingSequenceINT(BoS):
    lengths = [S.size()[0] for S in BoS]
    maxLength = max(lengths)
    newBoS = []
    #ss = [S.size() for S in BoS]
    #print ss
    for i in range(len(BoS)):
        #print BoS[i].size()
        if lengths[i] < maxLength:
            if torch.cuda.is_available():
                newBoS.append(torch.cat([BoS[i], Variable(torch.LongTensor([0] * (maxLength-lengths[i])), requires_grad = False).cuda().view(-1)]))
            else:
                newBoS.append(torch.cat([BoS[i], Variable(torch.LongTensor([0] * (maxLength-lengths[i])), requires_grad = False).view(-1)]))
        else:
            newBoS.append(BoS[i])
    newBoS = torch.stack(newBoS)
    return newBoS

def Indexing_2DINT(operations, h):
    ZERO = Variable(torch.LongTensor([0]), requires_grad = False)
    if torch.cuda.is_available():
        ZERO = ZERO.cuda()
    ZERO = ZERO.view(-1)
    #print ZERO
    #print h.size()
    
    bidx = 0
    newH = []
    for actions in operations:
        h_i = []
        pt = -1
        for oper, id in actions:
            if (oper == 2):
                pt += 1
            if (pt < 0):
                h_i.append(ZERO)
            else:
                h_i.append(h[bidx,pt])
            #print bidx, pt
        h_i = torch.cat(h_i, dim = 0)
        newH.append(h_i)
        bidx += 1
    return PaddingSequenceINT(newH)

def numpy2torch(x):
    x = Variable(torch.from_numpy(x)) if x is not None else None
    if torch.cuda.is_available():
        x = x.cuda() if x is not None else None
    return x

def torch2numpy(x):
    x = x.cpu() if x is not None else None
    x = x.data.numpy() if x is not None else None
    return x

def topKIndexes(dist, K, gamma = 0):
    if (K >= dist.shape[0]):
        K = dist.shape[0]
    indexes = np.argpartition(dist, -K)[-K:]
    if gamma == 0:
        return indexes.flatten()
    threshold = np.min(dist[indexes]) - gamma
    indexes = np.argwhere(dist > threshold)
    return indexes.flatten()

def translate(seq, Vocab, doc):
    sent = ''
    for item in seq:
        if item[0] == 1:
            break
        if item[0] == 0:
            att = int(item[1].argmax())
            word = doc[att]
        else:
            word = Vocab.i2w[item[0]]
            if '#' in word:
                cands = []
                Index = 0
                for token in doc:
                    if remove_digits(token).lower() == word:
                        cands += [Index]
                    Index += 1
                if Index != 0:
                    bestScore = 0.0
                    bestCand = 0
                    for cand in cands:
                        if float(item[1][cand]) > bestScore:
                            bestScore = float(item[1][cand])
                            bestCand = cand
                    word = doc[bestCand]
        word = word.lower()
        sent += word + ' '
    return sent

UNK, BOS, EOS = 0, 1, 2
OTHER, REDUCE, NT, GEN = 0, 1, 2, 3
ROOT, REDUCE_L, REDUCE_R = 0, 1, 2

def translateParse(seq, Vocab, doc, NT_Map):
    sent = ''
    for item in seq:
        if item[0] == 0:
            word = 'OTHER'
        elif item[0] == 1:
            word = ')'
        elif (2 <= item[0]) and (item[0] <30):
            word = '(' + NT_Map['i2w'][item[0]-2]
        elif item[0] == 30 + UNK:
            att = int(item[1].argmax())
            word = doc[att]
            word.lower()
        else:
            word = Vocab.i2w[item[0] - 30]
            if '#' in word:
                cands = []
                Index = 0
                for token in doc:
                    if remove_digits(token).lower() == word:
                        cands += [Index]
                    Index += 1
                if Index != 0:
                    bestScore = 0.0
                    bestCand = 0
                    for cand in cands:
                        if float(item[1][cand]) > bestScore:
                            bestScore = float(item[1][cand])
                            bestCand = cand
                    word = doc[bestCand]
            word = word.lower()
        sent += word + ' '
    return sent

def translateParse_De(seq, Vocab, doc):
    sent = ''
    for item in seq:
        if item[0] == 0:
            word = 'ROOT'
        elif item[0] == 1:
            word = '<--'
        elif item[0] == 2:
            word = '-->'
        elif item[0] == 3 + UNK:
            att = int(item[1].argmax())
            word = doc[att]
            word.lower()
        else:
            word = Vocab.i2w[item[0] - 3]
            if '#' in word:
                cands = []
                Index = 0
                for token in doc:
                    if remove_digits(token).lower() == word:
                        cands += [Index]
                    Index += 1
                if Index != 0:
                    bestScore = 0.0
                    bestCand = 0
                    for cand in cands:
                        if float(item[1][cand]) > bestScore:
                            bestScore = float(item[1][cand])
                            bestCand = cand
                    word = doc[bestCand]
            word = word.lower()       
        sent += word + ' '
    return sent

def unpack2Dto3D(M, lengths):
    maxLength = max(lengths)
    n_batch = len(lengths)
    dim = M.size()[1]
    start, end = 0, 0
    seqs = []
    for i in range(n_batch):
        end = start + lengths[i]
        seq = M[start:end,:]
        if lengths[i] != maxLength:
            seq = torch.cat([seq, Variable(torch.zeros((maxLength - lengths[i], dim)), requires_grad = False).cuda()], dim = 0)
        seqs.append(seq)
        start = end
    return torch.stack(seqs)

def packList2Dto3D(seqs):
    lengths = [seq.size()[0] for seq in seqs]
    maxLength = max(lengths)
    result = []
    for seq in seqs:
        if seq.size()[0] == maxLength:
            result.append(seq)
        else:
            result.append(torch.cat([seq, Variable(torch.zeros((maxLength - seq.size()[0], seq.size()[1])), requires_grad = False).cuda()], dim = 0))
    return torch.stack(result)

def indexSelect_2D_3D(T, idx):
    h, w, d = T.size()
    idx_ = Variable(torch.arange(h).long())
    if torch.cuda.is_available():
        idx_ = idx_.cuda()
    idx_ = idx_* w + idx
    result = torch.index_select(T.view(-1, d), 0, idx_)
    return result
    
def getRelations(text):
    actions = text.lower().split()
    stack = []
    relations = []
    for act in actions:
        if act == '-->':
            modifier = stack.pop()
            head = stack.pop()
            relations.append((head, modifier))
            stack.append(head)
        elif act == '<--':
            head = stack.pop()
            modifier = stack.pop()
            relations.append((head, modifier))
            stack.append(head)
        else:
            stack.append(act)
    return set(relations)

def actions2text(actions, Vocab):
    text = ""
    for act in actions:
        if act[0] == 1:
            text += "<-- "
        elif act[0] == 2:
            text += "--> "
        else:
            text += Vocab[act[1]] + " "
    return text

def rmReduce(text):
    result = text.replace(' -->','')
    result = result.replace(' <--','')
    return result
    

def reversedSet(X):
    if len(X) == 0:
        return X
    a, b = zip(*list(X))
    return set(zip(b, a))
    
def compRelations(A, B):
    '''
        A is hyp
        B is ref
    '''
    #nA = reversedSet(A)
    nB = reversedSet(B)
    #AA = A | nA
    BB = B | nB
    
    match = len(A&B)
    match_ = len(A&BB)
    
    return match, match_, len(A), len(B) 