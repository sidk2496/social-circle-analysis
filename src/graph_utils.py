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
        self.nodes = deepcopy(nodes)
        self.edges = deepcopy(edges)
        self.adjlist = deepcopy(adjlist)
        self.circles = deepcopy(circles)
        self.update_graph()
        self.is_converge = False

    def add(self, node_id, circle_id, temp_node, temp_circle):
        circle = self.circles[circle_id]
        similarity = np.min(self.sim_matrix[node_id, list(circle.members)])
        prob = similarity # sigmoid(50 * (avg_similarity - 0.5))
        if prob > 0.5:
            temp_node.membership.add(circle_id)
            temp_circle.members.add(node_id)

    def union(self, u_id, v_id):
        u = self.nodes[u_id]
        v = self.nodes[v_id]
        temp_u = deepcopy(self.nodes[u_id])
        temp_v = deepcopy(self.nodes[v_id])
        temp_circles = {}
        u_symdiff_v = u.membership ^ v.membership
        for circle_id in u_symdiff_v:
            temp_circles[circle_id] = deepcopy(self.circles[circle_id])

        u_diff_v = u.membership - v.membership
        for circle_id in u_diff_v:
            self.add(v.id, circle_id, temp_v, temp_circles[circle_id])
        v_diff_u = v.membership - u.membership
        for circle_id in v_diff_u:
            self.add(u.id, circle_id, temp_u, temp_circles[circle_id])

        self.nodes[u_id] = deepcopy(temp_u)
        self.nodes[v_id] = deepcopy(temp_v)

        for circle_id in temp_circles.keys():
            self.circles[circle_id] = deepcopy(temp_circles[circle_id])

    def circle_formation(self):
        for edge in self.edges:
            self.union(edge.u_id, edge.v_id)

    def label_propagation(self, alpha):
        temp_nodes = deepcopy(self.nodes)
        for node_id, node in enumerate(self.nodes):
            neighbor_indices = set({})
            for circle_id in node.membership:
                neighbor_indices.update(list(self.circles[circle_id].members))

            weights = self.sim_matrix[node_id, list(neighbor_indices)]
            weights /= weights.sum()
            one_indices = node.attributes == 1
            neighbor_attributes = np.vstack([self.nodes[nei_id].attributes for nei_id in neighbor_indices])
            new_attributes = np.matmul(weights, neighbor_attributes)
            new_attributes[one_indices] = 1
            temp_nodes[node_id].attributes = alpha * node.attributes + (1 - alpha) * new_attributes

            # if node_id in self.adjlist.keys():
            #     neighbor_attributes = [self.nodes[neighbor_id].attributes for neighbor_id in self.adjlist[node_id]]
            #     neighbor_attributes = np.array(neighbor_attributes)
            #     neighbor_avg = np.average(neighbor_attributes, axis=0)
            #     temp_nodes[node_id].attributes = alpha * node.attributes + (1 - alpha) * neighbor_avg
        self.check_convergence(self.nodes, temp_nodes)
        self.nodes = deepcopy(temp_nodes)

    def dissolve_circles(self, threshold):
        circles = self.circles.values()
        temp_circles = deepcopy(self.circles)
        delete_circle_ids = []
        for iter, circle1 in enumerate(circles):
            for circle2 in list(circles)[iter + 1:]:
                iou, i, u = IoU_score(circle1, circle2)
                if iou > threshold:
                	# Doesn't matter if we iterate through new_circles or temp_circles
                	# They will always be the same for circle2 i.e.,
                	# initial value of new_circles[circle2.id] (if becoming circle2 for the first time)
                	# or empty circle (if becoming circle2 again)
                    for node_id in circle2.members:
                        node = self.nodes[node_id]
                        node.membership.remove(circle2.id)
                        node.membership.add(circle1.id)
                        temp_circles[circle1.id].members.add(node_id)
                    circle2.members.clear()
                    temp_circles[circle2.id].members.clear()
                    delete_circle_ids.append(circle2.id)

        for circle_id in delete_circle_ids:
            del self.circles[circle_id]
            del temp_circles[circle_id]

        self.circles = deepcopy(temp_circles)

    def update_graph(self):
        attribute_matrix = np.vstack([node.attributes for node in self.nodes])
        self.sim_matrix = cosine_similarity(attribute_matrix)
        for edge in self.edges:
            u_id = edge.u_id
            v_id = edge.v_id
            edge.w = self.sim_matrix[u_id, v_id]
        np.fill_diagonal(self.sim_matrix, 1)

    def post_clustering(self):
        delete_circle_ids = []
        for circle in self.circles.values():
            if circle.id not in self.adjlist.keys():
                delete_circle_ids.append(circle.id)

        for circle_id in delete_circle_ids:
            del self.circles[circle_id]

    def check_convergence(self, old_nodes, new_nodes):
        old_attributes = np.vstack([node.attributes for node in old_nodes])
        new_attributes = np.vstack([node.attributes for node in new_nodes])
        diff = np.absolute(old_attributes - new_attributes)
        avg_diff = np.average(diff)
        if (avg_diff < 0.00001):
            self.is_converge = True

