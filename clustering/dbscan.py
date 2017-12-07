#-*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN as sk_DBSCAN
from .clustering import Clustering

class DBSCAN(Clustering):
    """docstring for DBSCAN."""
    def __init__(self, data, eps = 0.5, min_samples = 5, metric = 'euclidean',
                 algorithm = 'auto', leaf_size = 30, p = None, n_jobs = 1):
        super(DBSCAN, self).__init__()
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs


    def execute(self):
        """Constroi o modelo de clusterizacao."""
        self.model = sk_DBSCAN(eps = self.eps,
                               min_samples = self.min_samples,
                               metric = self.metric,
                               algorithm = self.algorithm,
                               leaf_size = self.leaf_size,
                               p = self.p,
                               n_jobs = self.n_jobs).fit(self.data)

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
