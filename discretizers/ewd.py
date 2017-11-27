# -*- coding: utf-8 -*-
import numpy as np

class EWD(object):
    """docstring for EWD"""
    def __init__(self, data, n_tracks):
        super(EWD, self).__init__()
        self.data = data
        self.n_tracks = n_tracks
        self.n_elements = len(self.data)
        self.n_attrs = len(self.data[0])
        self.edges = []


    def discretize(self):
        """Inicia o processo de discretizacao."""
        discrete_data = []

        for attr_index in range(self.n_attrs):
            edges = self.__calc_edges(attr_index)
            self.edges.append(edges)
            discrete_data.append(self.__fit(attr_index, edges))

        # Armazena os dados discretizados
        self.discrete_data = np.transpose(np.array(discrete_data))


    def __calc_edges(self, attr_index):
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



    def __fit(self, attr_index, edges):
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


    @property
    def discrete_data_(self):
        """Retorna os dados discretizados"""
        return self.discrete_data

    @property
    def edges_(self):
        """Retorna os pontos de corte dos atributos"""
        return self.edges
