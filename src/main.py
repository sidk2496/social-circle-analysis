from graph_utils import *
from input_utils import *
import pickle
import argparse

def main(args):
    input_filename = args.inp
    alpha = args.alpha
    iter = args.iter
    threshold = args.threshold

    data_dir = '../../data/facebook/processed/'
    with open(data_dir + input_filename, 'rb') as input_file:
        graph = pickle.load(input_file)

    nodes = graph['nodes']
    edges = graph['edges']

    # Initialize 1 circle for each node
    circles = [Circle(node_id, {node_id}) for node_id in nodes.keys()]
    circles = dict(zip(nodes.keys(), circles))

    while iter:
        edges.sort(key=lambda edge: edge.w)
        circle_formation()
        label_propagation(nodes, adjlist, )






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=float, help='IoU threshold')
    parser.add_argument('--a', type=float, help='alpha for updating label values')
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--iter', type=float, help='max number of iterations')
    args = parser.parse_args()
    main(args)