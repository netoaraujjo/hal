from sklearn.cluster import KMeans as sk_KMeans
from .clustering import Clustering

class KMeans(Clustering):
	"""docstring for KMeans."""
	def __init__(self, n_clusters, init_method, n_init, data):
		super(KMeans, self).__init__()
		self.n_clusters = n_clusters
		self.init = init_method
		self.n_init = n_init
		self.data = data

	def execute(self):
		"""Constroi o modelo de clusterizacao."""
		self.model = sk_KMeans(n_clusters = self.n_clusters,
		                        init = self.init,
								n_init = self.n_init).fit(self.data)

		self.clusters = super().make_clusters(self.data, self.model.labels_)

	@property
	def labels_(self):
		"""Retorna os labels dos elementos do dataset."""
		return self.model.labels_

	@property
	def clusters_(self):
		"""Retorna um dicionaro onde os indices dos grupos sao as chaves."""
		return self.clusters

	@property
	def model_(self):
		"""Retorna o modelo de agrupamento."""
		return self.model_
