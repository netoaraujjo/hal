#-*- coding: utf-8 -*-

import sys
from iomanager import IOManager
import numpy as np
from sklearn.cluster import KMeans
from labeling.mra import MRA
from discretizers.ewd import EWD

def main(argv):
	# print('Bem vindo ao projeto HAL!')

	dataset = IOManager.read_file(argv[1], True)
	dataset['data'] = np.array(dataset['data'])


	# Discretizar dados
	ewd = EWD(dataset['data'], 3)
	ewd.discretize()


	# Executa o KMeans para agrupamento
	kmeans = KMeans(n_clusters = 3).fit(dataset['data'])


	# Constrói os grupos em dicionários, sendo os índices dos grupos as chaves do dicionário
	clusters = {}
	original_clusters = {} # armazena os clusters com valores originais
	for c in range(len(kmeans.labels_)):
		cluster_index = str(kmeans.labels_[c])

		if cluster_index in clusters.keys():
			clusters[cluster_index] = np.vstack((clusters[cluster_index], ewd.discrete_data_[c]))
			original_clusters[cluster_index] = np.vstack((original_clusters[cluster_index], dataset['data'][c]))
		else:
			clusters[cluster_index] = np.array(ewd.discrete_data_[c])
			original_clusters[cluster_index] = dataset['data'][c]


	# Executa o MLP para rotulação
	mra = MRA(original_clusters, clusters, dataset['attributes'], 5, ewd.edges_).execute()

if __name__ == '__main__':
	main(sys.argv)