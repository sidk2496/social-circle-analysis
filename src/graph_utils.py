import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy import unravel_index

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def IoU_score(circle1, circle2):
    I = len(circle1.nodes.intersection(circle2.nodes))
    U = len(circle1.nodes.union(circle2.nodes))
    return 0 if U == 0 else I / U

class Node:
    def __init__(self, id, attributes, membership):
        self.id = id
        self.membership = membership
        self.attributes = attributes

class Circle:
    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes
        self.num_nodes = len(nodes)

class Edge:
    def __init__(self, u_id, v_id):
        self.u_id = u_id
        self.v_id = v_id
        self.w = 0

class Graph:
    def __init__(self, nodes, edges, adjlist, circles):
        self.nodes = nodes
        self.edges = edges
        self.adjlist = adjlist
        self.circles = circles
        self.update_similarities()

    def randomized_add(self, node_id, circle_id):
        node = self.nodes[node_id]
        circle = self.circles[circle_id]
        avg_similarity = np.average(self.sim_matrix[node_id, list(circle.nodes)])
        prob = sigmoid(50 * (avg_similarity - 0.5))
        flip = np.random.binomial(1, prob, 1)
        # print("Prob of adding node {0} to circle {1}={2}".format(node_id, circle_id, prob))
        # print("Add or dont add: {0}".format(flip))
        if flip == 1:
        	# print("############# added ########" + str(flip))
        	node.membership.add(circle.id)
        	circle.nodes.add(node.id)
        	if (circle_id == 0):
        		print("{0} nodes in circle 0".format(len(circle.nodes)))

    def union(self, u_id, v_id):
        u = self.nodes[u_id]
        v = self.nodes[v_id]
        u_diff_v = u.membership.difference(v.membership)
        for circle_id in u_diff_v:
            self.randomized_add(v.id, circle_id)

        v_diff_u = v.membership.difference(u.membership)
        for circle_id in v_diff_u:
            self.randomized_add(u.id, circle_id)

    def circle_formation(self):
        for edge in self.edges:
            self.union(edge.u_id, edge.v_id)

    def label_propagation(self, alpha):
        nodes_temp = self.nodes.copy()
        for node_id, node in enumerate(self.nodes):
        	if node_id in self.adjlist.keys():
	            neighbor_attributes = [self.nodes[neighbor_id].attributes for neighbor_id in self.adjlist[node_id]]
	            neighbor_attributes = np.array(neighbor_attributes)
	            neighbor_avg = np.average(neighbor_attributes, axis=0)
	            nodes_temp[node_id].attributes = alpha * node.attributes + (1 - alpha) * neighbor_avg
        self.nodes = nodes_temp

    def dissolve_circles(self, threshold):
        circles = self.circles.values()
        delete_circle_ids = []
        for iter, circle1 in enumerate(circles):
            for circle2 in list(circles)[iter + 1: ]:
                iou = IoU_score(circle1, circle2)
                if iou > threshold:
                	print('thresh:' + str(th))
                    for node_id in circle2.nodes:
                        node = self.nodes[node_id]
                        node.membership.remove(circle2.id)
                        node.membership.add(circle1.id)
                        circle1.nodes.add(node_id)
                    circle2.nodes.clear()
                    delete_circle_ids.append(circle2.id)

        for circle_id in delete_circle_ids:
        	del self.circles[circle_id]

    def update_similarities(self):
        attribute_matrix = np.vstack([node.attributes for node in self.nodes])
        self.sim_matrix = cosine_similarity(attribute_matrix)
        for edge in self.edges:
            u_id = edge.u_id
            v_id = edge.v_id
            edge.w = self.sim_matrix[u_id, v_id]