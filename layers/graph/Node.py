class Node(object):
    def __init__(self, id , emb, hidden):
        self.id = id
        self.children = None
        self.emb = emb
        self.hidden = hidden
        self.parent = -1
    
    def setChildren(self, children):
        self.children = children
    
    def setParent(self, parent):
        self.parent = parent
        
    def setHidden(self, hidden):
        self.hidden = hidden