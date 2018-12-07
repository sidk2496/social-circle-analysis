from graph_utils import *
from scipy.optimize import linear_sum_assignment
import pickle

def BER_F1(egonet_pred, ego_id, rev_mapping):
	universe = set([rev_mapping[node_id] for node_id in egonet_pred.adjlist.keys()])
	pred_circles = list(egonet_pred.circles.values())
	true_circles = []

	with open('../../data/facebook/' + ego_id + '.circles') as file:
		lines = file.read().splitlines()
		for line in lines:
			members = set(map(int, line.split('\t')[1: ]))
			true_circles.append(members)

	num_pred_circles = len(egonet_pred.circles)
	num_true_circles = len(true_circles)
	BER_matrix = np.zeros((num_pred_circles, num_true_circles))
	F1_score_matrix = np.zeros((num_pred_circles, num_true_circles))

	for row, pred_circle in enumerate(pred_circles):
		pred_members = set([rev_mapping[node_id] for node_id in pred_circle.members])
		for col, true_circle in enumerate(true_circles):
			TP = len(pred_members & true_circle)
			TN = len(universe - (pred_members | true_circle))
			FP = len(pred_members - true_circle)
			FN = len(true_circle - pred_members)
			err0 =  FP / (FP + TN) 
			err1 =  FN / (FN + TP)
			BER = 0.5 * (err0 + err1)
			BER_matrix[row, col] = BER
			precision = TP / (TP + FP)
			recall = TP / (TP + FN)
			F1_score = 2 * precision * recall / (precision + recall + 1e-6)
			F1_score_matrix[row, col] = F1_score

	row_indices, col_indices = linear_sum_assignment(BER_matrix)
	avg_BER_score = 1 - np.average(BER_matrix[row_indices, col_indices])
	row_indices, col_indices = linear_sum_assignment(1 - F1_score_matrix)
	avg_F1_score = np.average(F1_score_matrix[row_indices, col_indices])
	# print(BER_matrix.shape)
	# print(BER_matrix[row_indices, col_indices])
	# print(row_indices)
	# for row in row_indices:
	# 	for n in pred_circles[row].members:
	# 		print(rev_mapping[n])

	return avg_BER_score, avg_F1_score