# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np

class Discretization(ABC):
    """docstring for Discretization."""
    def __init__(self):
        super(Discretization, self).__init__()

    @property
    @abstractmethod
    def discrete_data_(self):
        """Retorna os dados discretizados."""
        pass

    @property
    @abstractmethod
    def edges_(self):
        """Retorna os pontos de corte dos atributos."""
        pass

    @property
    @abstractmethod
    def discrete_clusters_(self):
        """Retorna os clusters com valores discretos."""
        pass

    def make_discrete_clusters(self, discrete_data, labels):
        """Constroi os grupos em dicionarios, sendo os indices dos grupos as
        chaves do dicionario."""
        discrete_clusters = {} # armazena os clusters com valores originais
        for c in range(len(labels)):
            cluster_index = str(labels[c])
            if cluster_index in discrete_clusters.keys():
                discrete_clusters[cluster_index] = np.vstack((discrete_clusters[cluster_index],
                                                              discrete_data[c]))
            else:
                discrete_clusters[cluster_index] = np.array(discrete_data[c])

        return discrete_clusters
