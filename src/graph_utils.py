def similarity(u, v):
    return

class Node:
    def __init__(self, id, attributes, membership=None):
        self.id = id
        self.membership = membership
        self.attributes = attributes

class Circle:
    def __init__(self, id, nodes):
        self.id = id
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
        for circle_id in u_diff_v:
            randomized_add(circles[circle_id], v)

    v_diff_u = u.membership.difference(v.membership)
    if len(v_diff_u) != 0:
        for circle_id in v_diff_u:
            randomized_add(circles[circle_id], u)

def circle_formation(nodes, edges, circles):
    for edge in edges:
        union(edge.u, edge.v, circles)
    dissolve_circles(nodes, circles)

def label_propagation(nodes, edges, adjlist, alpha):
    nodes_temp = nodes.copy()
    for id, node in nodes.items():
        neighbor_attributes = [neigbhor.attributes for neighbor in adjlist[id]]
        neighbor_attributes = np.array(neighbor_attributes)
        neighbor_avg = np.avg(neighbor_attributes, axis=0)
        nodes_temp[id].attributes = alpha * node.attributes + (1 - alpha) * neighbor_avg
    nodes = nodes_temp

def IoU_score(circle1, circle2):
    I = len(circle1.nodes.intersection(circle2.nodes))
    U = len(circle1.nodes.union(circle2.nodes))
    return I / U

def dissolve_circles(nodes, circles, threshold):
    circle_ids = circles.keys()
    for iter, circle_id1 in enumerate(circle_ids):
        for circle_id2 in circle_ids[iter + 1: ]:
            iou = IoU_score(circles[circle_id1], circles[circle_id2])
            if iou > threshold:
                for node_id in circles[circle_id2].nodes:
                    nodes[node_id].membership.remove(circle_id2)
                    nodes[node_id].membership.add(circle_id1)
                del circles[circle_id2]