# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

class MRA(object):
	"""docstring for MRA"""
	def __init__(self, clusters, attributes, variacao):
		super(MRA, self).__init__()
		self.clusters = clusters
		self.attributes = attributes
		self.variacao = variacao
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

		print(self.relevancies)
		print()


		self.select_attributes()


	def select_attributes(self):
		"""Seleciona os atributos de acordo com o parâmetro de variação"""
		for cluster, relevance in self.relevancies.items():
			label = {}
			maximo = max(relevance.values())
			for attr, rel in relevance.items():
				if rel >= maximo - self.variacao:
					label[attr] = rel
			self.labels[cluster] = label

		print(self.labels)
		print()
		self.calc_tracks()



	def calc_tracks(self):
		for cluster in self.clusters:
			for attr_index in len(self.attributes):
				pass



	def calc_accuracy(self):
		"""Calcula a acuracia da rotulação"""
		pass