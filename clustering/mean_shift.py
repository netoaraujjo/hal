#-*- coding: utf-8 -*-
from sklearn.cluster import MeanShift as sk_MeanShift
from .clustering import Clustering

class MeanShift(Clustering):
    """docstring for MeanShift."""
    def __init__(self, data, bandwidth = None, seeds = None, bin_seeding = False,
                 min_bin_freq = 1, cluster_all = True, n_jobs = 1):
        super(MeanShift, self).__init__()
        self.data = data
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.n_jobs = n_jobs


    def execute(self):
        """Constroi o modelo de clusterizacao."""
        self.model = sk_MeanShift(bandwidth = self.bandwidth,
                                  seeds = self.seeds,
                                  bin_seeding = self.bin_seeding,
                                  min_bin_freq = self.min_bin_freq,
                                  cluster_all = self.cluster_all,
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
