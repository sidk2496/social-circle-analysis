import pickle

file_count = 2
with open('../../data/facebook/processed/data_' + str(file_count) + '.pkl', 'rb') as data_file:
	data = pickle.load(data_file)
