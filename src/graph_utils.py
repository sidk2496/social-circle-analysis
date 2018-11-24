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

def randomized_add(circle, v):
    similarities = map(lambda x: similarity(v, x), circle.nodes)
    avg_similarity = sum(similarities) / len(similarities)


def union(u, v, circles):
    u_diff_v = u.membership.difference(v.membership)
    if len(u_diff_v) != 0:
        for circle_idx in u_diff_v:
            randomized_add(circles[circle_idx], v)

    v_diff_u = u.membership.difference(v.membership)
    if len(v_diff_u) != 0:
        for circle_idx in v_diff_u:
            randomized_add(circles[circle_idx], u)


def circle_formation(nodes, edges, circles):
    for edge in edges:
        union(edge.u, edge.v, circles)
    dissolve_circles(nodes, circles)

def label_propagation(nodes, edges, circles):


def IoU_score(circle1, circle2):
    I = len(circle1.nodes.intersection(circle2.nodes))
    U = len(circle1.nodes.union(circle2.nodes))
    return I / U

def dissolve_circles(nodes, circles):
    circle_ids = circles.keys()
    for i, idx1 in enumerate(circle_ids):
        for idx2 in circle_ids[i + 1: ]:
            iou = IoU_score(circles[idx1], circles[idx2])
            if iou > threshold:
                for node_idx in circles[idx2].nodes:
                    nodes[node_idx].membership.remove(idx2)
                del circles[idx2]






