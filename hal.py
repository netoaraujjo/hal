#-*- coding: utf-8 -*-

import sys
from iomanager import IOManager
import numpy as np
from sklearn.cluster import KMeans

def main(argv):
	print('Bem vindo ao projeto HAL!')

	dataset = IOManager.read_file(argv[1], True)
	dataset['data'] = np.array(dataset['data'])

	kmeans = KMeans(n_clusters = 3, ).fit(dataset['data'])

	clusters = {}

	for c in range(len(kmeans.labels_)):
		cluster_index = str(kmeans.labels_[c])

		if cluster_index in clusters.keys():
			clusters[cluster_index] = np.vstack((clusters[cluster_index], dataset['data'][c]))
		else:
			clusters[cluster_index] = np.array(dataset['data'][c])


if __name__ == '__main__':
	main(sys.argv)