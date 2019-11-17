def getBatchVocab(x, options):
    vocab_LVT = set(range(0,options['network']['Pred_tok']['n_out']))
    for l in x:
        vocab_LVT = vocab_LVT.union(set(l))
    vocab_LVT = list(vocab_LVT)
    vocab_LVT = sorted(list(vocab_LVT))
    dict_LVT = {}
    Index = 0
    for w in vocab_LVT:
        dict_LVT[w] = Index
        Index += 1
    return vocab_LVT, dict_LVT

def get_pointer(x, dict):
    result = []
    for l in x:
        result.append([dict[w] for w in l])
    return result

def sharpLVT(x, dict):
    result = []
    for l in x:
        temp = []
        for w in l:
            if w in dict:
                temp.append(dict[w])
            else:
                temp.append(0)
        result.append(temp)
    return result

def sharpOper(x, dict):
    result = []
    for l in x:
        temp = []
        for oper, id in l:
            if (oper == 2):
                if id in dict:
                    temp.append((2, dict[id]))
                else:
                    temp.append((2,0))
            else:
                temp.append((oper, id))
        result.append(temp)
    return result

def LVT(source, target, operations, options):
    batch_vocab, batch_dict = getBatchVocab(source, options)
    batch_target = sharpLVT(target, batch_dict)
    batch_oper = sharpOper(operations, batch_dict)
    pointer = get_pointer(source, batch_dict)
    return batch_vocab, pointer, batch_target, batch_oper 