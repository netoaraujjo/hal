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
        # discrete_data = np.zeros((self.n_elements, self.n_attrs))
        discrete_data = None

        for attr_index in range(self.n_attrs):
            edges = self.__calc_edges(attr_index)
            if discrete_data is None:
                discrete_data = self.__fit(attr_index, edges).reshape(self.n_elements,)
            else:
                discrete_data = np.hstack((discrete_data, self.__fit(attr_index, edges)))

        print('Discrete data:')
        print(discrete_data)
        print('ndim: ', discrete_data.ndim)
        print('shape: ', discrete_data.shape)
        print('type: ', type(discrete_data))
        print('len: ', len(discrete_data))
            


    def __calc_edges(self, attr_index):
        """Calcula os pontos de corte. Recebe o índice da coluna a ser discretizada"""
        col = self.data[:,attr_index]

        # print(col)
        # print('ndim: ', col.ndim)
        # print('shape: ', col.shape)

        maximo = max(col)
        minimo = min(col)
        largura = (maximo - minimo) / self.n_tracks

        # print('min: ', min(col), 'max: ', max(col), 'largura: ', largura)

        edges = []
        for cut_point in range(1, self.n_tracks):
            edges.append(minimo + cut_point * largura)

        print(edges)
        return edges



    def __fit(self, attr_index, edges):
        col = self.data[:,attr_index]
        
        print(col)
        print('ndim: ', col.ndim)
        print('shape: ', col.shape)
        print('type: ', type(col))
        print('len: ', len(col))

        discrete_col = np.zeros(self.n_elements,)

        for index, value in enumerate(col):
            for n, edge in enumerate(edges):
                if value <= edge:
                    discrete_col[index] = n + 1
                    break
                discrete_col[index] = self.n_tracks

        print('Discrete:')
        print(discrete_col)
        print('ndim: ', discrete_col.ndim)
        print('shape: ', discrete_col.shape)
        print('type: ', type(discrete_col))
        print('len: ', len(discrete_col))

        return discrete_col