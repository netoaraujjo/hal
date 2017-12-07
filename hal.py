#-*- coding: utf-8 -*-

import sys
from iomanager2 import IOManager2
import numpy as np
from labeling.mra import MRA
from discretizers.ewd import EWD
from discretizers.efd import EFD
from clustering.kmeans import KMeans
from clustering.birch import Birch
from clustering.dbscan import DBSCAN
from clustering.mean_shift import MeanShift
from clustering.affinity_propagation import AffinityPropagation

def main(argv):
	# print('Bem vindo ao projeto HAL!\n')

    # Le arquivo de entrada
	io = IOManager2()
	data = io.read_file(argv[1])
	print(" ESTATISTICAS")
	print(data.describe())

	# Executa o KMeans para agrupamento
	# clustering = KMeans(data.values, n_clusters = 3, init = 'random', n_init = 30, max_iter = 1000)
	# clustering = Birch(data.values, n_clusters = None)
	# clustering = DBSCAN(data.values)
	# clustering = MeanShift(data.values)
	clustering = AffinityPropagation(data.values)
	clustering.execute()

	# Discretizar dados
	ewd = EWD(data.values, 3)
	ewd.discretize(clustering.labels_)

	# efd = EFD(data.values, 3)
	# efd.discretize(clustering.labels_)

	# Executa o MLP para rotulacao
	mra = MRA(clustering.clusters_, ewd.discrete_clusters_, len(data), data.columns, 5,
			  ewd.edges_)
	mra.execute()
	print(mra.report_)

if __name__ == '__main__':
	main(sys.argv)
