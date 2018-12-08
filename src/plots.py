from graph_utils import *
from input_utils import *
import pickle
import argparse
from metrics_utils import metrics
import matplotlib
from matplotlib import pyplot as plt

def main(args):
    thresholds = np.arange(0.1, 1.1, 0.1)
    alpha = 0.99
    input_filename = 'egonet_' + args.inp + '.pkl'
    iterations = args.iter

    data_dir = '../../data/facebook/processed/'

    BER_scores = []
    F1_scores = []
    F1_circ_scores = []
    
    for threshold in thresholds:
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
            if it % 10 == 0:
                print('Iteration %d' % it)
        egonet.post_clustering()
        BER_score, F1_score, F1_circ = metrics(egonet, args.inp, rev_mapping)
        BER_scores.append(BER_score)
        F1_scores.append(F1_score)
        F1_circ_scores.append(F1_circ)

        np.save('f1_scores_{0}.npy'.format(args.inp), F1_scores)
        np.save('ber_scores_{0}.npy'.format(args.inp), BER_scores)

    font = {'size': 16}
    matplotlib.rc('font', **font)
    plt.figure(1)
    plt.plot(thresholds, F1_scores)
    plt.xlabel('IoU Threshold')

    plt.plot(thresholds, BER_scores)
    plt.xlabel('IoU Threshold')

    plt.plot(thresholds, F1_circ_scores)
    plt.xlabel('IoU Threshold')

    plt.legend(['F1 score', '1 - BER', 'F1 circles'])
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, help='alpha for updating label values')
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--iter', type=float, help='max number of iterations')
    args = parser.parse_args()
    main(args)