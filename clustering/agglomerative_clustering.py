#-*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import AgglomerativeClustering as sk_AgglomerativeClustering
from sklearn.externals.joblib import Memory
from .clustering import Clustering

class AgglomerativeClustering(Clustering):
    """docstring for AgglomerativeClustering."""
    def __init__(self, data, n_clusters = 2, affinity = 'euclidean',
                 memory = Memory(cachedir = None), connectivity = None,
                 compute_full_tree = 'auto', linkage = 'ward',
                 pooling_func = np.mean):
        super(AgglomerativeClustering, self).__init__()
        self.data = data
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.pooling_func = pooling_func



    def execute(self):
        """Constroi o modelo de clusterizacao."""
        self.model = sk_AgglomerativeClustering(n_clusters = self.n_clusters,
                                        affinity = self.affinity,
                                        memory = self.memory,
                                        connectivity = self.connectivity,
                                        compute_full_tree = self.compute_full_tree,
                                        linkage = self.linkage,
                                        pooling_func = self.pooling_func).fit(self.data)

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
        return self.model
