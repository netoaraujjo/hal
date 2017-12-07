from sklearn.cluster import KMeans as sk_KMeans
from .clustering import Clustering

class KMeans(Clustering):
	"""docstring for KMeans."""
	def __init__(self, data, n_clusters = 8, init = 'k-means++', n_init = 10,
				 max_iter = 300, tol = 0.0001, precompute_distances = 'auto',
				 verbose = 0, random_state = None, copy_x = True, n_jobs = 1,
				 algorithm = 'auto'):
		super(KMeans, self).__init__()
		self.data = data
		self.n_clusters = n_clusters
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.precompute_distances = precompute_distances
		self.verbose = verbose
		self.random_state = random_state
		self.copy_x = copy_x
		self.n_jobs = n_jobs
		self.algorithm = algorithm


	def execute(self):
		"""Constroi o modelo de clusterizacao."""
		self.model = sk_KMeans(n_clusters = self.n_clusters,
		                       init = self.init,
							   n_init = self.n_init,
							   max_iter = self.max_iter,
							   tol = self.tol,
							   precompute_distances = self.precompute_distances,
							   verbose = self.verbose,
							   random_state = self.random_state,
							   copy_x = self.copy_x,
							   n_jobs = self.n_jobs,
							   algorithm = self.algorithm).fit(self.data)

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
