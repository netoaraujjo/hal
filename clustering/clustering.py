#-*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np

class Clustering(ABC):
    """docstring for Clustering."""
    def __init__(self):
        super(Clustering, self).__init__()

    @abstractmethod
    def execute(self):
        """Executa o processo de clusterizacao."""
        pass

    @property
    @abstractmethod
    def labels_(self):
        """Retorna os labels dos clusters."""
        pass

    @property
    @abstractmethod
    def clusters_(self):
        """Retorna os elementos do data set com os labels."""
        pass

    @property
    @abstractmethod
    def model_(self):
        """Retorna o modelo utilizado na clusterizacao."""
        pass

    def make_clusters(self, data, labels):
        """
        Constrói os grupos em dicionários, sendo os índices dos grupos as
        chaves do dicionário
        """
        clusters = {}
        for c in range(len(labels)):
            cluster_index = str(labels[c])
            if cluster_index in clusters.keys():
                clusters[cluster_index] = np.vstack((clusters[cluster_index], data[c]))
            else:
                clusters[cluster_index] = data[c]

        return clusters
