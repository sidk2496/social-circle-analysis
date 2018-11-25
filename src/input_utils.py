import numpy as np
from graph_utils import Node, Edge
import pickle
import os

def parse_feat_file(filename):
    with open(data_dir + filename, 'r') as file:
        nodes = {}
        for line in file:
            line = line.rstrip()
            tokens = line.split(' ')
            id = int(tokens[0])
            attributes = np.array(list(map(int, tokens[1: ])))
            nodes[id] = Node(id, attributes, {id})
    return nodes

def parse_edge_file(filename):
    with open(data_dir + filename, 'r') as file:
        edges = []
        adjlist = {}
        for line in file:
            line = line.rstrip()
            tokens = line.split(' ')
            u_id = int(tokens[0])
            v_id = int(tokens[1])
            if adjlist.get(u_id, None) is None:
                adjlist[u_id] = []
            adjlist[u_id].append(v_id)
            if adjlist.get(v_id, None) is None:
                adjlist[v_id] = []
            adjlist[v_id].append(u_id)
            edges.append(Edge(u_id, v_id))
    return edges, adjlist

data_dir = '../../data/facebook/'
filenames = os.listdir(data_dir)

feat_files = [filename for filename in filenames if '.feat' in filename and 'names' not in filename]
egofeat_files = [filename for filename in filenames if '.egofeat' in filename]
edge_files = [filename for filename in filenames if '.edges' in filename]

ego_networks = {}
for filename in feat_files:
    ego_id = int(filename.split('.')[0])
    ego_network = {}
    ego_network['ego_id'] = ego_id
    ego_network['nodes'] = parse_feat_file(filename)
    ego_networks[ego_id] = ego_network
    
for filename in egofeat_files:
    ego_id = int(filename.split('.')[0])
    with open(data_dir + filename, 'r') as file:
        line = file.readline().rstrip()
        tokens = line.split(' ')
        attributes = np.array(list(map(int, tokens)))
    ego_networks[ego_id]['nodes'][ego_id] = Node(ego_id, attributes, {id})

for filename in edge_files:
    ego_id = int(filename.split('.')[0])
    ego_networks[ego_id]['edges'], ego_networks[ego_id]['adjlist'] = parse_edge_file(filename)

count = 1
for _, ego_network in ego_networks.items():
    with open(data_dir + 'processed/egonet_' + str(count) + '.pkl', 'wb') as ego_network_file:
        pickle.dump(ego_network, ego_network_file)
    count += 1