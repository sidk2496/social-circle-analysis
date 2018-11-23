import numpy as np
from graph_utils import Node
import os

def parse_feat_file(filename):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
        nodes = []
        for line in lines:
            tokens = line.split(' ')
            idx = int(tokens[0])
            attributes = list(map(int, tokens[1: ]))
            nodes.append(Node(idx, attributes))
    return nodes

def parse_edge_file(filename):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            tokens = line.split(' ')
            u =

data_dir = '../../facebook/'
filenames = os.listdir(data_dir)

feat_files = [filename for filename in filenames if '.feat' in filename]
egofeat_files = [filename for filename in filenames if '.egofeat' in filename]
edge_files = [filename for filename in filenames if '.edges' in filename]

nodes = {}
for filename in feat_files:
    nodes[filename.split('.')[0]] = parse_feat_file(filename)

nodes = {}
for filename in feat_files:
    nodes[filename.split('.')[0]] = parse_feat_file(filename)

