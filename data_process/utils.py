def Index2Word(Index, Vocab):
    return Vocab.i2w[Index]

def Word2Index(Word, Vocab):
    if (not Word in Vocab.w2i):
        if hasattr(Vocab, 'i2e'):
            Word = '<unk>'
        else:
            Word = '[UNK]'
    return Vocab.w2i[Word]

def Sentence2ListOfWord(sentence):
    listOfWord = sentence.split()
    return listOfWord

def ListOfWord2ListOfIndex(listOfWord, Vocab):
    listOfIndex = []
    for w in listOfWord:
        listOfIndex.append(Word2Index(w, Vocab))
    return listOfIndex
    
def Sentence2ListOfIndex(sentence, Vocab):
    return ListOfWord2ListOfIndex(Sentence2ListOfWord(sentence),Vocab)
                        