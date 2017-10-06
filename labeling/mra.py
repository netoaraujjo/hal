# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

class MRA(object):
	"""docstring for MRA"""
	def __init__(self, clusters, attributes, variacao, edges):
		super(MRA, self).__init__()
		self.clusters = clusters
		self.attributes = attributes
		self.variacao = variacao
		self.edges = edges
		self.relevancies = {}
		self.labels = {}
	

	def execute(self):

		# percorre cada cluster
		for cluster in self.clusters:
			relevance = {} # Armazena a relevância de cada atributo para o cluster atual

			# print('Cluster %s | Qtd. elementos: %d' % (cluster, len(self.clusters[cluster])))

			# percorre cada um dos atributos
			for col, attribute in enumerate(self.attributes):

				# 60% dos elementos para treino
				qtd_elementos_treino = round(len(self.clusters[cluster]) * 0.6)
				
				dados_treino = self.clusters[cluster][:qtd_elementos_treino,:]
				dados_teste = self.clusters[cluster][qtd_elementos_treino:,:]

				# separa o atributo avaliado dos dados de treino
				X_treino = np.hstack((dados_treino[:, :col], dados_treino[:, col+1:]))
				y_treino = dados_treino[:, col]

				# separa o atributo avaliado dos dados de teste
				X_teste = np.hstack((dados_teste[:, :col], dados_teste[:, col+1:]))
				y_teste = dados_teste[:, col]

				mlp = MLPClassifier(hidden_layer_sizes = (10,), max_iter = 2000)
				mlp.fit(X_treino, y_treino)

				predictions = mlp.predict(X_teste)

				relevance[attribute] = accuracy_score(y_teste, predictions) * 100
				self.relevancies[cluster] = relevance

		most_rel = self.select_attributes()
		self.calc_frequency(most_rel)


	def select_attributes(self):
		"""Seleciona os atributos mais relevantes de acordo com o parâmetro de variação"""
		most_rel = {}
		for cluster, relevance in self.relevancies.items():
			most_rel_attr = {} # atributos mais relevantes para o cluster atual
			maximo = max(relevance.values())
			for attr, rel in relevance.items():
				if rel >= maximo - self.variacao:
					most_rel_attr[attr] = rel
			most_rel[cluster] = most_rel_attr

		print(most_rel)
		return most_rel



	def calc_frequency(self, most_rel):
		# print(self.clusters)
		most_freq = {}
		for cluster, rel_attr in most_rel.items():
			most_freq_value = {}
			for attr in rel_attr.keys():
				attr_index = self.attributes.index(attr) # indice da coluna com valores do atributo
				values = (list(self.clusters[cluster][:,attr_index]))
				unique_values = list(set(values))
				frequency = []
				for value in unique_values:
					frequency.append(values.count(value))

				most_freq_value[attr] = unique_values[frequency.index(max(frequency))]

			most_freq[cluster] = most_freq_value

		print(most_freq)
		return most_freq


	def calc_accuracy(self):
		"""Calcula a acuracia da rotulação"""
		pass