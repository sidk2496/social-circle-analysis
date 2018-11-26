import numpy as np
from graph_utils import Node, Edge
import pickle
import os

def parse_feat_file(filename):
    with open(data_dir + filename, 'r') as file:
        nodes = []
        mapping = {}
        rev_mapping = {}
        lines = file.read().splitlines()
        for i, line in enumerate(lines):
            tokens = line.split(' ')
            node_id = int(tokens[0])
            attributes = np.array(list(map(int, tokens[1: ])))
            mapping[node_id] = i
            rev_mapping[i] = node_id
            nodes.append(Node(i, attributes, {i}))
    return nodes, mapping, rev_mapping

def parse_edge_file(filename, mapping):
    with open(data_dir + filename, 'r') as file:
        edges = []
        adjlist = {}
        lines = file.read().splitlines()
        for line in lines:
            tokens = line.split(' ')
            u_id = mapping[int(tokens[0])]
            v_id = mapping[int(tokens[1])]
            if u_id not in adjlist.keys():
                adjlist[u_id] = []
            adjlist[u_id].append(v_id)
            if v_id not in adjlist.keys():
                adjlist[v_id] = []
            adjlist[v_id].append(u_id)
            edges.append(Edge(u_id, v_id))
    return edges, adjlist

data_dir = '../../data/facebook/'
filenames = os.listdir(data_dir)

feat_files = [filename for filename in filenames if '.feat' in filename and 'names' not in filename]
edge_files = [filename for filename in filenames if '.edges' in filename]

egonets = {}
for filename in feat_files:
    ego_id = int(filename.split('.')[0])
    egonet = {}
    egonet['ego_id'] = ego_id
    egonet['nodes'], egonet['map'], egonet['rev_map'] = parse_feat_file(filename)
    egonets[ego_id] = egonet

for filename in edge_files:
    ego_id = int(filename.split('.')[0])
    egonet = egonets[ego_id]
    mapping = egonet['map']
    egonet['edges'], egonet['adjlist'] = parse_edge_file(filename, mapping)

for _, egonet in egonets.items():
    with open(data_dir + 'processed/egonet_' + str(egonet['ego_id']) + '.pkl', 'wb') as egonet_file:
        pickle.dump(egonet, egonet_file)