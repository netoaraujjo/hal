#-*- coding: utf-8 -*-
from sklearn.cluster import AffinityPropagation as sk_AffinityPropagation
from .clustering import Clustering

class AffinityPropagation(Clustering):
    """docstring for AffinityPropagation."""
    def __init__(self, data, damping = 0.5, max_iter = 200, convergence_iter = 15,
                 copy = True, preference = None, affinity = 'euclidean',
                 verbose = False):
        super(Clustering, self).__init__()
        self.data = data
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.preference = preference
        self.affinity = affinity
        self.verbose = verbose


    def execute(self):
        """Constroi o modelo de clusterizacao."""
        self.model = sk_AffinityPropagation(damping = self.damping,
                                  max_iter = self.max_iter,
                                  convergence_iter = self.convergence_iter,
                                  copy = self.copy,
                                  preference = self.preference,
                                  affinity = self.affinity,
                                  verbose = self.verbose).fit(self.data)

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
