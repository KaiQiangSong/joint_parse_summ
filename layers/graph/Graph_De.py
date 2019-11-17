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
from layers.utils import *
from .Node import Node
        
class Graph_De(object):
    ROOT, REDUCE_L, REDUCE_R, GEN = 0, 1, 2, 3
    
    def __init__(self, node, buffer_size = 512):
        self.node = node
        self.buffer_size = buffer_size
        self.N = 0
        self.Nodes = []
        self.Stacks = []
        self.nStack = []
        self.nG = 0
    
    def isEmpty(self):
        return self.N == 0
    
    def addNode(self, emb, hidden):
        newNode = Node(self.N, emb, hidden)
        self.Nodes.append(newNode)
        self.N += 1
        return newNode.id
    
    def setParent(self, parent, children):
        '''
            parent: an integer index
            children: a list of indexes
        '''
        for idx in children:
            self.Nodes[idx].setParent(parent)
        self.Nodes[parent].setChildren(children)
 
        
    def setHidden(self, ids, hiddens):
        N = len(ids)
        for i in range(N):
            self.Nodes[ids[i]].setHidden(hiddens[i])
    
    @staticmethod
    def buildGraph(node, operation, action_emb, length):
        G = Graph_De(node)
        stack = []
        for i in range(length):
            oper = operation[i]
            #print oper
            if (oper == Graph_De.ROOT) or (oper == Graph_De.GEN):
                id = G.addNode(action_emb[i], action_emb[i])
                stack.append(id)
                
            elif oper == Graph_De.REDUCE_L:
                head = stack.pop()
                modifier = stack.pop()
                id = G.addNode(G.Nodes[head].emb, None)
                G.setParent(id, [head, modifier])
                stack.append(id)
                
            elif oper == Graph_De.REDUCE_R:
                modifier = stack.pop()
                head = stack.pop()
                id = G.addNode(G.Nodes[head].emb, None)
                G.setParent(id, [head, modifier])
                stack.append(id)
                    
            G.Stacks.append(copy.deepcopy(stack))
            if G.nStack == []:
                G.nStack.append(1)
            else:
                G.nStack[0] += 1
        return G
    
    @staticmethod
    def buildGraphs(node, operation, action_emb, lengths):
        G = Graph_De(node)
        N = len(lengths)
        for i in range(N):
            tG = Graph_De.buildGraph(node, operation[i], action_emb[i], lengths[i])
            if G.isEmpty():
                G = tG
            else:
                G.Merge(tG)
        return G
    
    def Merge(self, G):
        for i in range(G.N):
            G.Nodes[i].id += self.N
            if (G.Nodes[i].parent >= 0):
                G.Nodes[i].parent += self.N
            if (G.Nodes[i].children is not None):
                for j in range(len(G.Nodes[i].children)):
                    G.Nodes[i].children[j] += self.N
        for i in range(len(G.Stacks)):
            for j in range(len(G.Stacks[i])):
                G.Stacks[i][j] += self.N
        
        self.N += G.N
        self.nG += 1
        self.Nodes += G.Nodes
        self.Stacks += G.Stacks
        self.nStack += G.nStack
    
    def toComputationalList(self):
        ind = [0] * self.N
        head, tail, queue, last = 0, 0, [], []
        for node in self.Nodes:
            if node.parent >=0:
                ind[node.parent] += 1
        
        for i in range(self.N):
            if ind[i] == 0:
                queue.append(i)
                last.append(-1)
                tail += 1
        
        while head < tail:
            cur = queue[head]
            parent = self.Nodes[cur].parent
            if parent >= 0:
                ind[parent] -= 1
                if ind[parent] == 0:
                    queue.append(parent)
                    last.append(head)
                    tail += 1
            head += 1
        
        return queue, last
    
    def partition(self):
        queue, last = self.toComputationalList()
        n = len(queue)
        
        buffer, buffers = [], []
        st = 0
        for i in range(n):
            if last[i] != -1:
                if buffer == []:
                    st = i
                if (last[i] >= st) or (i - st + 1 > self.buffer_size):
                    buffers.append(buffer)
                    buffer = []
                    st = i
                buffer.append(queue[i])
        
        if buffer != []:
            buffers.append(buffer)
        
        buffer_stacks = []
        cnt = 0
        for i in range(len(self.nStack)):
            stacks = self.Stacks[cnt:cnt + self.nStack[i]]
            buffer_stacks.append(stacks)
            cnt += self.nStack[i]
        
        return buffers, buffer_stacks
    
    def calc(self):
        buffers, buffer_stacks = self.partition()
        for buffer in buffers:
            
            pairs = torch.stack([torch.cat([self.Nodes[j].hidden for j in self.Nodes[i].children]) for i in buffer])
            nodeStates = self.node(pairs)
            self.setHidden(buffer, nodeStates)
            
        batch_size = len(buffer_stacks)
        stack_emb = []
        len_stacks = []
        for i in range(batch_size):
            stacks = buffer_stacks[i]
            n_stack = len(stacks)
            len_stack = []
            for j in range(n_stack):
                temp = []
                len_stack.append(len(stacks[j]))
                for k in range(len(stacks[j])):
                    temp.append(self.Nodes[stacks[j][k]].hidden)
                    
                stack_emb.append(torch.stack(temp))
            len_stacks.append(len_stack)
        
        return packList2Dto3D(stack_emb), len_stacks
    