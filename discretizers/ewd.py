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



    def discretize(self):
        discrete_data = []

        for attr_index in range(self.n_attrs):
            edges = self.__calc_edges(attr_index)
            discrete_data.append(self.__fit(attr_index, edges))

        return np.transpose(np.array(discrete_data))
            


    def __calc_edges(self, attr_index):
        """Calcula os pontos de corte. Recebe o índice da coluna a ser discretizada"""
        col = self.data[:,attr_index]

        maximo = max(col)
        minimo = min(col)
        largura = (maximo - minimo) / self.n_tracks

        edges = [minimo]
        for cut_point in range(1, self.n_tracks):
            edges.append(minimo + cut_point * largura)

        edges.append(maximo)
        return edges



    def __fit(self, attr_index, edges):
        col = self.data[:,attr_index]

        discrete_col = []

        for index, value in enumerate(col):
            for n, edge in enumerate(edges[1:len(edges)-1]):
                if value <= edge:
                    discrete_col.append(n+1)
                    break

            if len(discrete_col) == index:
                discrete_col.append(self.n_tracks)

        # for index, value in enumerate(col):
        #     for n in range(1,len(edges)):
        #         if value <= edges[n]:
        #             discrete_col.append(n)
        #             break

        return discrete_col