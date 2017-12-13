# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np

class Discretization(ABC):
    """docstring for Discretization."""
    def __init__(self):
        super(Discretization, self).__init__()


    @abstractmethod
    def calc_edges(self, attr_index):
        """Calcula os pontos de corte. Recebe o índice da coluna a ser
        discretizada.
        attr_index eh o indice da coluna (atributo)
        """
        pass

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

            # Remover em caso de problema com keys
            if labels[c] < 10:
                cluster_index = "0" + cluster_index

            if cluster_index in discrete_clusters.keys():
                discrete_clusters[cluster_index] = np.vstack((discrete_clusters[cluster_index],
                                                              discrete_data[c]))
            else:
                discrete_clusters[cluster_index] = np.array(discrete_data[c])

        return discrete_clusters


    def fit(self, data, n_tracks, attr_index, edges):
        """Calcula os valores discretos para todos os elementos em um dado
        atributo.
        attr_index eh o indice da coluna (atributo) cujos valores serão
        discretizados.
        edges sao os pontos de corte para o atributo a ser discretizado.
        """
        col = self.data[:,attr_index]

        discrete_col = []

        for index, value in enumerate(col):
            for n, edge in enumerate(edges[1:len(edges)-1]):
                if value <= edge:
                    discrete_col.append(n+1)
                    break

            if len(discrete_col) == index:
                discrete_col.append(self.n_tracks)

        # Retorna a coluna (atributos) com valores discretos
        return discrete_col
