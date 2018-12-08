from graph_utils import *
from input_utils import *
import pickle
import argparse
from metrics_utils import metrics

def main(args):
    threshold = args.t
    alpha = args.a
    input_filename = 'egonet_' + args.inp + '.pkl'
    iterations = args.iter

    data_dir = '../../data/facebook/processed/'
    with open(data_dir + input_filename, 'rb') as input_file:
        egonet = pickle.load(input_file)

    nodes = egonet['nodes']
    edges = egonet['edges']
    adjlist = egonet['adjlist']
    rev_mapping = egonet['rev_map']

    circles = [Circle(node_id, {node_id}) for node_id in range(len(nodes))]

    circles = dict(zip(range(len(nodes)), circles))
    egonet = Graph(nodes, edges, adjlist, circles)

    it = 0
    while it < iterations:
        it += 1
        egonet.edges.sort(key=lambda edge: edge.w, reverse=True)
        egonet.circle_formation()
        egonet.dissolve_circles(threshold)
        egonet.label_propagation(alpha)
        egonet.update_graph()
        if egonet.is_converge:
            break
        print('Iteration %d' % it)

    egonet.post_clustering()
    print('Number of circles: ' + str(len(egonet.circles)))

    for circle_id, circle in egonet.circles.items():
        print('circle' + str(rev_mapping[circle_id]))
        for node_id in circle.members:
            print(rev_mapping[node_id], end=' ')
        print('\n')

    BER_score, F1_score, F1_circ = metrics(egonet, args.inp, rev_mapping)
    print('BER score: %f\nF1 score: %f\nF1 circles: %f' % (BER_score, F1_score, F1_circ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=float, help='IoU threshold')
    parser.add_argument('--a', type=float, help='alpha for updating label values')
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--iter', type=float, help='max number of iterations')
    args = parser.parse_args()
    main(args)