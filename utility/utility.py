import torch
'''
import torch.nn as nn
import torch.nn.functional as functional
import torch.autograd as autograd
from torch.autograd import Variable
'''

import numpy as np
import json, re, shutil
import pickle

# IO
def loadFromJson(filename):
    f = open(filename,'r',encoding = 'utf-8')
    data = json.load(f,strict = False)
    f.close()
    return data

def saveToJson(filename, data):
    f = open(filename,'w',encoding = 'utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True

def saveToPKL(filename, data):
    with open(filename, 'wb')as f:
        pickle.dump(data, f)
    return

def loadFromPKL(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return data

def writeFile(filename, massage):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(massage)
    return True

# Initializing
def zero_weights(n_in ,n_out = None):
    if (n_out == None):
        W = np.zeros(n_in)
    else:
        W = np.zeros(n_in, n_out)
    return W.astype('float32')

def orthogonal_weights(n_dim):
    W = np.random.randn(n_dim,n_dim)
    u, _ , _ =np.linalg.svd(W)
    return u.astype('float32')

def random_weights(n_in,n_out, scale = None):
    if scale is None:
        scale = np.sqrt(2.0 / (n_in + n_out))
    W = scale * np.random.randn(n_in,n_out)
    return W.astype('float32')

def remove_digits(parse):
    return re.sub(r'\d', '#', parse)

def save_check_point(state, is_best, fileName = './model/checkpoint.pth.tar'):
    torch.save(state, fileName)
    if is_best:
        shutil.copyfile(fileName, './model/model_best.pth.tar')
        shutil.copyfile(fileName, './model/model_best_epoch_'+str(state['epoch'])+'.pth.tar')

def RougeTrick(parse):
    '''
    parse = re.sub(r'#','XXX',parse)
    parse = re.sub(r'XXX-','XXXYYY',parse)
    parse = re.sub(r'-XXX','YYYXXX',parse)
    parse = re.sub(r'XXX.','XXXWWW',parse)
    parse = re.sub(r'.XXX','WWWXXX',parse)
    parse = re.sub(r'<unk>','ZZZZZ',parse)
    '''
    parse = re.sub(r'#','T',parse)
    parse = re.sub(r'T-','TD',parse)
    parse = re.sub(r'-T','DT',parse)
    parse = re.sub(r'TX.','TB',parse)
    parse = re.sub(r'.T','BT',parse)
    parse = re.sub(r'<unk>','UNK',parse)
    
    return parse
