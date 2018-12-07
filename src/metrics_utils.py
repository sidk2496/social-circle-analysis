from scipy.optimize import linear_sum_assignment
import pickle
import numpy as np

def IoU_score(circle1_members, circle2_members):
    I = len(circle1_members & circle2_members)
    U = len(circle1_members | circle2_members)
    return 0 if U == 0 else I / U

def metrics(egonet_pred, ego_id, rev_mapping):
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

	# max BER-score assignment
	row_indices, col_indices = linear_sum_assignment(BER_matrix)
	avg_BER_score = 1 - np.average(BER_matrix[row_indices, col_indices])

	# max F1-score assignment
	row_indices, col_indices = linear_sum_assignment(1 - F1_score_matrix)
	avg_F1_score = np.average(F1_score_matrix[row_indices, col_indices])

	TP_circ = 0
	FP_circ = 0
	FN_circ = 0
	num_match = len(row_indices)

	for i in range(num_match):
		pred_members = set([rev_mapping[node_id] for node_id in pred_circles[row_indices[i]].members])
		true_circle = true_circles[col_indices[i]]
		iou = IoU_score(pred_members, true_circle)
		TP_circ += iou
		FP_circ += 1 - iou
		FN_circ += 1 - iou

	for row, pred_circle in enumerate(pred_circles):
		if row not in row_indices:
			max_IoU = 0
			pred_members = set([rev_mapping[node_id] for node_id in pred_circle.members])
			for true_circle in true_circles:
				max_IoU = max(max_IoU, IoU_score(pred_members, true_circle))
			FP_circ += 1 - max_IoU

	for col, true_circle in enumerate(true_circles):
		if col not in col_indices:
			max_IoU = 0
			for pred_circle in pred_circles:
				pred_members = set([rev_mapping[node_id] for node_id in pred_circle.members])
				max_IoU = max(max_IoU, IoU_score(pred_members, true_circle))
			FN_circ += 1 - max_IoU

	precision_circ = TP_circ / (TP_circ + FP_circ + 1e-6)
	recall_circ = TP_circ / (TP_circ + FN_circ + 1e-6)
	F1_circ = 2 * precision_circ * recall_circ / (precision_circ + recall_circ + 1e-6)
	# print(BER_matrix.shape)
	# print(BER_matrix[row_indices, col_indices])
	# print(row_indices)
	# for row in row_indices:
	# 	for n in pred_circles[row].members:
	# 		print(rev_mapping[n])

	return avg_BER_score, avg_F1_score, F1_circ