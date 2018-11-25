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
            attributes = np.array(map(int, tokens[1: ]))
            nodes[id] = Node(id, attributes, {id})
    return nodes

def parse_edge_file(filename, nodes):
    with open(data_dir + filename, 'r') as file:
        edges = []
        adjlist =  dict.fromkeys(nodes.keys(), [])
        for line in file:
            line = line.rstrip()
            tokens = line.split(' ')
            u = int(tokens[0])
            v = int(tokens[1])
            adjlist[u].append(nodes[v])
            adjlist[v].append(nodes[u])
            edges.append(Edge(nodes[u], nodes[v]))
    return edges, adjlist

data_dir = '../../data/facebook/'
filenames = os.listdir(data_dir)

feat_files = [filename for filename in filenames if '.feat' in filename and 'names' not in filename]
egofeat_files = [filename for filename in filenames if '.egofeat' in filename]
edge_files = [filename for filename in filenames if '.edges' in filename]

datasets = {}
for filename in feat_files:
    ego_id = int(filename.split('.')[0])
    dataset = {}
    dataset['ego_id'] = ego_id
    dataset['nodes'] = parse_feat_file(filename)
    datasets[ego_id] = dataset
    
for filename in egofeat_files:
    ego_id = int(filename.split('.')[0])
    with open(data_dir + filename, 'r') as file:
        line = file.readline()
        line = line.rstrip()
        tokens = line.split(' ')
        attributes = np.array(map(int, tokens))
    datasets[ego_id]['nodes'][ego_id] = Node(ego_id, attributes, {id})

for filename in edge_files:
    ego_id = int(filename.split('.')[0])
    datasets[ego_id]['edges'], datasets[ego_id]['adjlist'] = parse_edge_file(filename, datasets[ego_id]['nodes'])

count_dataset = 1
for _, dataset in datasets.items():
    with open(data_dir + 'processed/data_' + str(count_dataset) + '.pkl', 'wb') as data_file:
        pickle.dump(dataset, data_file)
    count_dataset += 1