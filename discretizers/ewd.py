# -*- coding: utf-8 -*-
from .discretization import Discretization
import numpy as np

class EWD(Discretization):
    """docstring for EWD"""
    def __init__(self, data, n_tracks):
        super(EWD, self).__init__()
        self.data = data
        self.n_tracks = n_tracks
        self.n_elements = len(self.data)
        self.n_attrs = len(self.data[0])
        self.edges = []


    def discretize(self, labels):
        """Inicia o processo de discretizacao."""
        discrete_data = []

        for attr_index in range(self.n_attrs):
            edges = self.calc_edges(attr_index)
            self.edges.append(edges)
            discrete_data.append(super().fit(self.data, self.n_tracks, attr_index, edges))

        # Armazena os dados discretizados
        self.discrete_data = np.transpose(np.array(discrete_data))

        self.discrete_clusters = super().make_discrete_clusters(self.discrete_data, labels)


    def calc_edges(self, attr_index):
        """Calcula os pontos de corte. Recebe o índice da coluna a ser
        discretizada.
        attr_index eh o indice da coluna (atributo)
        """
        col = self.data[:,attr_index]

        maximo = max(col)
        minimo = min(col)
        largura = (maximo - minimo) / self.n_tracks

        edges = [minimo]
        for cut_point in range(1, self.n_tracks):
            edges.append(minimo + cut_point * largura)

        edges.append(maximo)

        # Retorna uma lista contendo os pontos de corte do intervalo de valores
        # do atributo
        return edges


    @property
    def discrete_data_(self):
        """Retorna os dados discretizados."""
        return self.discrete_data

    @property
    def edges_(self):
        """Retorna os pontos de corte dos atributos."""
        return self.edges

    @property
    def discrete_clusters_(self):
        """Retorna os clusters com valores discretos."""
        return self.discrete_clusters
