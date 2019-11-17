#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch

#==============================#
#    Class/Layer Part Import   #
#==============================#

class OpMapping_De(object):
    ROOT, REDUCE_L, REDUCE_R, GEN = 0, 1, 2, 3
    '''
        Dependency Parsing
        
        ACT:
        Action: Int[0..3]
            OTHER, REDUCE_L, REDUCE_R, GEN
        
        OPI:
        Operation_Index: Int [0..3+|V|-1]
            0: OTHERE
            1: REDUCE_L
            2: REDUCE_R
            3..3+|V|-1: GEN X
            
        OPP
        Operation_Pair: Pair (Int, Int)
            first element: OTHER, REDUCE_L, REDUCE_R, GEN [0..3]
            second element:
                If GEN : [0..|V|]
        
        TOK
        Token: Int
    
        LVT
        Larget Vocabulary Trick Index: Int
    
        Ele: Element
        Seq: One Sequence
        Seqs: Sequences
        Tensor1D: Tensor
        Tensor2D: Tensor, Lengths
        Parse: Parse Tree
    
        ID List:
            0: Other
            1: Reduce
            2~29: NT X
            30~: GEN Y
    '''
    @staticmethod
    def OPPEle2OPIEle(ele):
        oper, id = ele
        if (oper < 3):
            return oper
        return id + 3
    
    @staticmethod
    def OPPSeq2OPISeq(seq):
        return [OpMapping_De.OPPEle2OPIEle(ele) for ele in seq]
    
    @staticmethod
    def OPPSeqs2OPISeqs(seqs):
        return [OpMapping_De.OPPSeq2OPISeq(seq) for seq in seqs]
    
    @staticmethod
    def OPIEle2OPPEle(ele):
        if (ele < 3):
            return (ele, 0)
        return (3, ele - 3)
    
    @staticmethod
    def OPISeq2OPPSeq(seq):
        return [OpMapping_De.OPIEle2OPPEle(ele) for ele in seq]
    
    @staticmethod
    def OPISeqs2OPPSeqs(seqs):
        return [OpMapping_De.OPISeq2OPPSeq(seq) for seq in seqs]
    
    @staticmethod
    def TOKEle2OPIEle(ele):
        return ele + 3
    
    @staticmethod
    def TOKSeq2OPISeq(seq):
        return [OpMapping_De.TOKEle2OPIEle(ele) for ele in seq]
    
    @staticmethod
    def TOKSeqs2OPISeqs(seqs):
        return [OpMapping_De.TOKSeq2OPISeq(seq) for seq in seqs]
    
    @staticmethod
    def TOKTensor2OPITensor(T):
        return T + 3
    
    @staticmethod
    def TOKTensor1D2OPITensor1D(T):
        return OpMapping_De.TOKTensor2OPITensor(T)
    
    @staticmethod
    def TOKTensor2D2OPITensor2D(T):
        return OpMapping_De.TOKTensor2OPITensor(T)
    
    @staticmethod
    def OPITensor2TOKTensor(T):
        TOK = torch.where(T>=3, T - 3, torch.zeros_like(T))
        return TOK
    
    @staticmethod
    def LVTSeq2TOKSeq(seq, Pool):
        res = []
        for ele in seq:
            if Pool[ele] >= 3:
                res.append(Pool[ele] - 3)
        return res
    
    @classmethod
    def LVTSeq2ParseSeq(cls, seq, Pool, Vocab):
        OPI_seq = [Pool[ele] for ele in seq]
        OPP_Seq = OpMapping_De.OPISeq2OPPSeq()
        res = []
        for oper, id in OPP_Seq:
            if oper == cls.OTHER:
                res.append('')
            elif oper == cls.REDUCE_L:
                res.append('REDUCE_L')
            elif oper == cls.REDUCE_R:
                res.append('REDUCE_R')
            elif oper == cls.GEN:
                res.append(Vocab.i2w[id])
        return res
    
    @staticmethod
    def OPIEle2ACTEle(ele):
        if ele >= 3:
            return 3
        return ele
    
    @staticmethod
    def OPISeq2ACTSeq(seq):
        return [OpMapping_De.OPIEle2ACTEle(ele) for ele in seq]
    
    @staticmethod
    def OPISeqs2ACTSeqs(seqs):
        return [OpMapping_De.OPISeq2ACTSeq(seq) for seq in seqs]
    
    @staticmethod
    def OPITensor2ACTTensor(T):
        ACT = torch.where(T >= 3, torch.ones_like(T) * 3, T)
        return ACT
    
    @staticmethod
    def OPITensor2ACT4Tensor(T):
        return OpMapping_De.OPITensor2ACTTensor(T)
    
    @staticmethod
    def OPITensor2IDTensor(T):
        return torch.where(T >= 3, T - 3, torch.zeros_like(T))

