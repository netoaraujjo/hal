#-*- coding: utf-8 -*-
from sklearn.cluster import Birch as sk_Birch
from .clustering import Clustering

class Birch(Clustering):
    """docstring for Birch."""
    def __init__(self, data, threshold = 0.5, branching_factor = 50,
                 n_clusters = 3, compute_labels = True, copy = True):
        super(Birch, self).__init__()
        self.data = data
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy


    def execute(self):
        """Constroi o modelo de clusterizacao."""
        self.model = sk_Birch(threshold = self.threshold,
                              branching_factor = self.branching_factor,
                              n_clusters = self.n_clusters,
                              compute_labels = self.compute_labels,
                              copy = self.copy).fit(self.data)

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
