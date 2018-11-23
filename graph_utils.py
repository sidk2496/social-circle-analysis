def similarity(u, v):
    return

class Node:
    def __init__(self, idx, attributes, membership=None):
        self.idx = idx
        self.membership = membership
        self.attributes = attributes

class Circle:
    def __init__(self, idx, nodes):
        self.idx = idx
        self.nodes = nodes
        self.num_nodes = len(nodes)

class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v
        self.w = similarity(u, v)


