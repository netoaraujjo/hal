#-*- coding: utf-8 -*-

import sys
from iomanager2 import IOManager2
import numpy as np
from sklearn.cluster import KMeans
from labeling.mra import MRA
from discretizers.ewd import EWD

def main(argv):
	# print('Bem vindo ao projeto HAL!\n')

	io = IOManager2()
	data = io.read_file(argv[1])
	print(data.describe())

	# Discretizar dados
	ewd = EWD(data.values, 3)
	ewd.discretize()


	# Executa o KMeans para agrupamento
	kmeans = KMeans(n_clusters = 3, init = 'random', n_init = 30).fit(data.values)
	print(kmeans.labels_)

	#####################################################
	# Comum para todos os alforitmos de agrupamento
	#####################################################

	# Constrói os grupos em dicionários, sendo os índices dos grupos as chaves do dicionário
	clusters = {}
	original_clusters = {} # armazena os clusters com valores originais
	for c in range(len(kmeans.labels_)):
		cluster_index = str(kmeans.labels_[c])
		if cluster_index in clusters.keys():
			clusters[cluster_index] = np.vstack((clusters[cluster_index], ewd.discrete_data_[c]))
			original_clusters[cluster_index] = np.vstack((original_clusters[cluster_index], data.values[c]))
		else:
			clusters[cluster_index] = np.array(ewd.discrete_data_[c])
			original_clusters[cluster_index] = data.values[c]

	#####################################################

	

	# Executa o MLP para rotulação
	# mra = MRA(original_clusters, clusters, dataset['attributes'], 5, ewd.edges_).execute()

if __name__ == '__main__':
	main(sys.argv)
