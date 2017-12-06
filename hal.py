#-*- coding: utf-8 -*-

import sys
from iomanager2 import IOManager2
import numpy as np
from sklearn.cluster import KMeans
from labeling.mra import MRA
from discretizers.ewd import EWD
from clustering.kmeans import KMeans

def main(argv):
	# print('Bem vindo ao projeto HAL!\n')

	io = IOManager2()
	data = io.read_file(argv[1])
	print(data.describe())

	# Executa o KMeans para agrupamento
	clustering = KMeans(3, 'random', 30, data.values)
	clustering.execute()

	# Discretizar dados
	ewd = EWD(data.values, 3)
	ewd.discretize(clustering.labels_)

	# Constroi os grupos em dicionarios, sendo os indices dos grupos as chaves do dicionario
	# discrete_clusters = {} # armazena os clusters com valores originais
	# for c in range(len(clustering.labels_)):
	# 	cluster_index = str(clustering.labels_[c])
	# 	if cluster_index in discrete_clusters.keys():
	# 		discrete_clusters[cluster_index] = np.vstack((discrete_clusters[cluster_index], ewd.discrete_data_[c]))
	# 	else:
	# 		discrete_clusters[cluster_index] = np.array(ewd.discrete_data_[c])

	# Executa o MLP para rotulacao
	mra = MRA(clustering.clusters_, ewd.discrete_clusters_, data.columns, 5, ewd.edges_).execute()

if __name__ == '__main__':
	main(sys.argv)
