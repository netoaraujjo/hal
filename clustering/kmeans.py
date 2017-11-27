from sklearn.cluster import KMeans

# Executa o KMeans para agrupamento
	kmeans = KMeans(n_clusters = 3, init = 'random', n_init = 30).fit(dataset['data'])