#==============================#
#    System Import             #
#==============================#

#==============================#
#    Platform Import           #
#==============================#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#==============================#
#    Class/Layer Part Import   #
#==============================#
from layers.basic import Attention, Linear, LSTM, MaskLSTM
from .Initializer import Initializer
from layers.utils import oneHot, unpack2Dto3D

class Decoder_deRNNG(nn.Module):
    ROOT, REDUCE_L, REDUCE_R, GEN = 0, 1, 2, 3
    def __init__(self, options):
        super(Decoder_deRNNG, self).__init__()
        # stack-LSTM
        self.stack_init = Initializer(options['stack_init'])
        self.stack = LSTM(options['stack'])
        
        # history-LSTM
        self.history_init = Initializer(options['history_init'])
        self.history = LSTM(options['history'])
        
        # buffer-LSTM
        self.buffer_init = Initializer(options['buffer_init'])
        self.buffer = MaskLSTM(options['buffer'])
        
        # Attention
        self.attention = Attention(options['attention'])
        
        # Prediction
        self.hidden_act = Linear(options['hidden_act'])
        self.hidden_tok = Linear(options['hidden_tok'])
        
        self.predict_act = Linear(options['predict_act'])
        self.predict_tok = Linear(options['predict_tok'])
        
        # Copy Mechanism
        self.switcher_copy = Linear(options['switcher_copy'])
        
    def forward(self, LVT, state_below, node, stack_emb = None, len_stacks = None, 
                operation = None, action_emb = None, action_mask = None, len_action = None, state_prev = None):
        '''
            state_prev:
                stack: current node in stack
                top: the top index for stack
        '''
        h_e, source_mask = state_below
        
        if state_prev is None:
            '''
                stack-LSTM Part
            '''
            h_d1_init, c_d1_init = self.stack_init(h_e, source_mask)
            
            # Generating the h_d_init and c_d_init for different stacks
            n_batches = len(len_stacks)
            lengths = [len(len_stack) for len_stack in len_stacks]
            idx = [i for i in range(n_batches) for j in range(lengths[i])]
            idx = Variable(torch.LongTensor(idx))
            if torch.cuda.is_available():
                idx = idx.cuda()
            idx_oneHot = oneHot(idx, n_batches)
            h_d1_init_calc = torch.mm(idx_oneHot, h_d1_init)
            c_d1_init_calc = torch.mm(idx_oneHot, c_d1_init)
            
            len_stacks_ = Variable(torch.LongTensor([l for len_stack in len_stacks for l in len_stack]))
            if torch.cuda.is_available():
                len_stacks_ = len_stacks_.cuda()
            
            
            sorted_len_stacks_, IDs = torch.sort(len_stacks_, descending = True)
            _, bIDs = torch.sort(IDs, descending = False)
            sorted_stack_emb = torch.index_select(stack_emb, 0, IDs)
            
            packed_stack_emb = pack_padded_sequence(sorted_stack_emb, [int(it) for it in list(sorted_len_stacks_)], batch_first = True)
        
            h_d1_init_calc = torch.index_select(h_d1_init_calc, 0, IDs)
            c_d1_init_calc = torch.index_select(c_d1_init_calc, 0, IDs)
            
            h_d1_calc, (ph_d1_calc, pc_d1_calc) = self.stack(packed_stack_emb, (h_d1_init_calc, c_d1_init_calc))
        
            ph_d1_calc = torch.index_select(ph_d1_calc, 1, bIDs)[0]
            
            h_d1 = unpack2Dto3D(ph_d1_calc, lengths)
            
            '''
                History-LSTM Part
            '''
            h_d2_init, c_d2_init = self.history_init(h_e, source_mask)
            
             
            sorted_len_action, IDs = torch.sort(len_action, descending = True)
            _, bIDs = torch.sort(IDs, descending = False)
            
            sorted_action_emb = torch.index_select(action_emb, 0, IDs)
            h_d2_init_calc = torch.index_select(h_d2_init, 0, IDs)
            c_d2_init_calc = torch.index_select(c_d2_init, 0, IDs)
            
            
            packed_action_emb = pack_padded_sequence(sorted_action_emb, [int(it) for it in list(sorted_len_action)], batch_first = True)
            
            h_d2_calc, (ph_d2_calc, pc_d2_calc) = self.history(packed_action_emb, (h_d2_init_calc, c_d2_init_calc))
            h_d2_calc = pad_packed_sequence(h_d2_calc, batch_first = True)[0]
            h_d2 = torch.index_select(h_d2_calc, 0, bIDs)
            
            '''
                Buffer-LSTM Part
            '''
            h_d3_init, c_d3_init = self.buffer_init(h_e, source_mask)
            h_d3, c_d3 = self.buffer(action_emb, action_mask, (h_d3_init, c_d3_init))
            
            
            #h_d = torch.cat([h_d1, h_d2], dim = 2)
            
            '''
                Attention
            '''
            alpha = self.attention(h_e, torch.cat([h_d1, h_d3], dim = 2), source_mask)
            context = torch.bmm(alpha, h_e)
            
            '''
                Prediction
            '''
            # Hidden Representations
            hidden_act = F.tanh(self.hidden_act(torch.cat([h_d1, h_d2, context], dim = 2)))
            hidden_tok = F.tanh(self.hidden_tok(torch.cat([h_d1, h_d3, context], dim = 2)))
            
            # Prediction of Action
            pred_act = F.softmax(self.predict_act(hidden_act), dim = 2)
            pred_tok = F.softmax(self.predict_tok(hidden_tok), dim = 2)
            
            switcher_copy = F.sigmoid(self.switcher_copy(torch.cat([h_d1, h_d3, context], dim = 2)))
            
            pool_size = LVT.size - 3
        
            if pool_size > pred_tok.size()[2]:
                if torch.cuda.is_available():
                    prob_vocab = switcher_copy * torch.cat([pred_tok, Variable(torch.zeros((pred_tok.size()[0], pred_tok.size()[1], pool_size - pred_tok.size()[2])), requires_grad = False).cuda()], dim = 2)
                else:
                    prob_vocab = switcher_copy * torch.cat([pred_tok, Variable(torch.zeros((pred_tok.size()[0], pred_tok.size()[1], pool_size - pred_tok.size()[2])), requires_grad = False)], dim = 2)
            else:
                prob_vocab = switcher_copy * pred_tok
            
            shp = LVT.Pointer.size()
            POINTER = torch.where(LVT.Pointer >= 3, LVT.Pointer - 3, torch.zeros_like(LVT.Pointer))
            pointer = oneHot(POINTER.view(-1), pool_size).view(shp[0], shp[1], pool_size)
            prob_copy = (1 - switcher_copy) * torch.bmm(alpha, pointer)
            prob = prob_vocab + prob_copy
        
            return [pred_act, prob, alpha, h_d1, h_d2, h_d3, hidden_act, hidden_tok], []
            
        stack, top, n_gen, (h_d2_prev, c_d2_prev), (h_d3_prev, c_d3_prev) = state_prev
        
        operation = operation[0][0]
        action_emb = action_emb[0][0]
        oper = int(operation)
        
        if (oper == self.ROOT):
            stack.append(action_emb)
            top += 1
            h_d2_prev, c_d2_prev = self.history_init(h_e, source_mask)
            h_d3_prev, c_d3_prev = self.buffer_init(h_e, source_mask)
            
        elif (oper ==self.GEN):
            stack.append(action_emb)
            top += 1
            n_gen += 1
            
        elif oper == self.REDUCE_L:
            head = stack.pop()
            modifier = stack.pop()
            top -= 2
            
            pair = torch.cat([head, modifier])
            hidden = node(pair[None, :])
            stack.append(hidden[0])
            top += 1
            
        elif oper == self.REDUCE_R:
            modifier = stack.pop()
            head = stack.pop()
            top -= 2
            
            pair = torch.cat([head, modifier])
            
            hidden = node(pair[None, :])
            stack.append(hidden[0])
            top += 1
        
        '''
            stack-LSTM Part
        '''
        h_d1_init, c_d1_init = self.stack_init(h_e, source_mask)
        
        stack_emb = torch.stack([stack[it] for it in range(len(stack))])[None,:,:]
        packed_stack_emb = pack_padded_sequence(stack_emb, [len(stack)], batch_first = True)
            
        h_d1, (ph_d1, pc_d1) = self.stack(packed_stack_emb, (h_d1_init, c_d1_init))
        h_d1 = ph_d1[0][None,:,:]
        
        '''
            history-LSTM Part
        '''
        
        packed_action_emb = pack_padded_sequence(action_emb[None, None, :], [int(it) for it in len_action.tolist()], batch_first = True)
        h_d2, (ph_d2, pc_d2) = self.history(packed_action_emb, (h_d2_prev, c_d2_prev))
        h_d2 = pad_packed_sequence(h_d2)[0]
        
        '''
            buffer-LSTM Part
        '''
        h_d3, c_d3 = self.buffer(action_emb[None,None,:], action_mask, (h_d3_prev, c_d3_prev))
        h_d3, c_d3 = h_d3[:,-1,:], c_d3[:,-1,:]
        h_d3_ = h_d3[:,None,:]
        
        '''
            Attention
        '''
        alpha = self.attention(h_e, torch.cat([h_d1, h_d3_], dim = 2), source_mask)
        context = torch.bmm(alpha, h_e)
        
        '''
            Prediction
        '''
        hidden_act = F.tanh(self.hidden_act(torch.cat([h_d1, h_d2, context], dim = 2)))
        hidden_tok = F.tanh(self.hidden_tok(torch.cat([h_d1, h_d3_, context], dim = 2)))
            
        pred_act = F.softmax(self.predict_act(hidden_act), dim = 2)
        pred_tok = F.softmax(self.predict_tok(hidden_tok), dim = 2)
        
        switcher_copy = F.sigmoid(self.switcher_copy(torch.cat([h_d1, h_d3_, context], dim = 2)))
        pool_size = LVT.size - 3
        
        if pool_size > pred_tok.size()[2]:
            if torch.cuda.is_available():
                prob_vocab = switcher_copy * torch.cat([pred_tok, Variable(torch.zeros((pred_tok.size()[0], pred_tok.size()[1], pool_size - pred_tok.size()[2])), requires_grad = False).cuda()], dim = 2)
            else:
                prob_vocab = switcher_copy * torch.cat([pred_tok, Variable(torch.zeros((pred_tok.size()[0], pred_tok.size()[1], pool_size - pred_tok.size()[2])), requires_grad = False)], dim = 2)
        else:
            prob_vocab = switcher_copy * pred_tok
            
        shp = LVT.Pointer.size()
        POINTER = torch.where(LVT.Pointer >= 3, LVT.Pointer - 3, torch.zeros_like(LVT.Pointer))
        pointer_ = oneHot(POINTER.view(-1), pool_size).view(shp[0], shp[1], pool_size)
        prob_copy = (1 - switcher_copy) * torch.bmm(alpha, pointer_)
        prob = prob_vocab + prob_copy
        
        
        return [pred_act, prob, alpha, h_d1, h_d2, h_d3, hidden_act, hidden_tok], [stack, top, n_gen, (ph_d2, pc_d2), (h_d3, c_d3) ]
    
    