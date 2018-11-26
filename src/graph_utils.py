import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def IoU_score(circle1, circle2):
    I = len(circle1.members.intersection(circle2.members))
    U = len(circle1.members.union(circle2.members))
    return 0 if U == 0 else I / U, I, U


class Node:
    def __init__(self, id, attributes, membership):
        self.id = id
        self.membership = membership
        self.attributes = attributes


class Circle:
    def __init__(self, id, members):
        self.id = id
        self.members = members


class Edge:
    def __init__(self, u_id, v_id):
        self.u_id = u_id
        self.v_id = v_id
        self.w = 0


class Graph:
    def __init__(self, nodes, edges, adjlist, circles):
        self.old_nodes = deepcopy(nodes)
        self.new_nodes = deepcopy(nodes)
        self.edges = deepcopy(edges)
        self.adjlist = deepcopy(adjlist)
        self.old_circles = deepcopy(circles)
        self.new_circles = deepcopy(circles)
        self.update_graph()

    def randomized_add(self, node_id, circle_id):
        new_node = self.new_nodes[node_id]
        old_circle = self.old_circles[circle_id]
        new_circle = self.new_circles[circle_id]
        avg_similarity = np.min(self.sim_matrix[node_id, list(old_circle.members)])
        prob = sigmoid(50 * (avg_similarity - 0.5))
        # prob = min(max(avg_similarity, 0), 1)
        flip = np.random.binomial(1, prob, 1)
        if flip == 1:
            new_node.membership.add(circle_id)
            new_circle.members.add(node_id)
            # if circle_id == 0:
            #     print("Randomized add added to the circle ")

    def union(self, u_id, v_id):
        u = self.old_nodes[u_id]
        v = self.old_nodes[v_id]
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
        nodes_temp = deepcopy(self.new_nodes)
        for node_id, node in enumerate(self.new_nodes):
            if node_id in self.adjlist.keys():
                neighbor_attributes = [self.new_nodes[neighbor_id].attributes for neighbor_id in self.adjlist[node_id]]
                neighbor_attributes = np.array(neighbor_attributes)
                neighbor_avg = np.average(neighbor_attributes, axis=0)
                nodes_temp[node_id].attributes = alpha * node.attributes + (1 - alpha) * neighbor_avg
        self.new_nodes = deepcopy(nodes_temp)

    def dissolve_circles(self, threshold):
        circles = self.new_circles.values()
        delete_circle_ids = []
        for iter, circle1 in enumerate(circles):
            for circle2 in list(circles)[iter + 1:]:
                iou, i, u = IoU_score(circle1, circle2)
                # print("Intersection: {0}, Union: {1}".format(i, u))
                if iou > threshold:

                    for node_id in circle2.members:
                        node = self.new_nodes[node_id]
                        node.membership.remove(circle2.id)
                        node.membership.add(circle1.id)
                        circle1.members.add(node_id)
                        # if circle1.id == 0:
                        #     print("Dissolve circles added to the circle")
                    circle2.members.clear()
                    self.new_circles[circle2.id].members.clear()
                    delete_circle_ids.append(circle2.id)

        for circle_id in delete_circle_ids:
            del self.new_circles[circle_id]

    def update_graph(self):
        attribute_matrix = np.vstack([node.attributes for node in self.new_nodes])
        self.sim_matrix = cosine_similarity(attribute_matrix)
        # print(self.sim_matrix[0, self.adjlist[0]])
        for edge in self.edges:
            u_id = edge.u_id
            v_id = edge.v_id
            edge.w = self.sim_matrix[u_id, v_id]
        self.old_nodes = deepcopy(self.new_nodes)
        self.old_circles = deepcopy(self.new_circles)
