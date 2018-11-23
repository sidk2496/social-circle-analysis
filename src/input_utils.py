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
            idx = int(tokens[0])
            attributes = list(map(int, tokens[1: ]))
            nodes[idx] = Node(idx, attributes, [idx])
    return nodes

def parse_edge_file(filename, nodes):
    with open(data_dir + filename, 'r') as file:
        edges = []
        for line in file:
            line = line.rstrip()
            tokens = line.split(' ')
            u = int(tokens[0])
            v = int(tokens[1])
            edges.append(Edge(nodes[u], nodes[v]))
    return edges

data_dir = '../../data/facebook/'
filenames = os.listdir(data_dir)

feat_files = [filename for filename in filenames if '.feat' in filename and 'names' not in filename]
egofeat_files = [filename for filename in filenames if '.egofeat' in filename]
edge_files = [filename for filename in filenames if '.edges' in filename]

datasets = {}
for filename in feat_files:
    ego_idx = int(filename.split('.')[0])
    dataset = {}
    dataset['ego_idx'] = ego_idx
    dataset['nodes'] = parse_feat_file(filename)
    datasets[ego_idx] = dataset
    
for filename in egofeat_files:
    ego_idx = int(filename.split('.')[0])
    with open(data_dir + filename, 'r') as file:
        line = file.readline()
        line = line.rstrip()
        tokens = line.split(' ')
        attributes = list(map(int, tokens))
    datasets[ego_idx]['nodes'][ego_idx] = Node(ego_idx, attributes, [idx])

for filename in edge_files:
    ego_idx = int(filename.split('.')[0])
    datasets[ego_idx]['edges'] = parse_edge_file(filename, datasets[ego_idx]['nodes'])

count_dataset = 1
for _, dataset in datasets.items():
    with open(data_dir + 'processed/data_' + str(count_dataset) + '.pkl', 'wb') as data_file:
        pickle.dump(dataset, data_file)
    count_dataset += 1