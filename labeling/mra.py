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
	

	def execute(self):
		# percorre cada cluster
		for cluster in self.clusters:
			print('Avaliando cluster %s' % cluster)

			# percorre cada um dos atributos
			for col, attribute in enumerate(self.attributes):
				print('Avaliando atributo %s' % attribute)

				# 60% dos elementos para treino
				qtd_elementos_treino = round(len(self.clusters[cluster]) * 0.6)
				
				dados_treino = self.clusters[cluster][:qtd_elementos_treino,:]
				dados_teste = self.clusters[cluster][qtd_elementos_treino:,:]

				X_treino = np.hstack((dados_treino[:, :col], dados_treino[:, col+1:]))
				y_treino = dados_treino[:, col]

				X_teste = np.hstack((dados_teste[:, :col], dados_teste[:, col+1:]))
				y_teste = dados_teste[:, col]
				# y_teste = y_teste.reshape(len(y_teste),1)

				# print(X_treino, X_treino.ndim, len(X_treino))
				# print(y_treino, y_treino.ndim, len(y_treino))

				mlp = MLPClassifier(hidden_layer_sizes = (10,))
				mlp.fit(X_treino, y_treino)
				# print("score treino: %f" % mlp.score(X_treino, y_treino))
				# print("score teste %f" % mlp.score(X_teste, y_teste))

				predictions = mlp.predict(X_teste)
				print(predictions)

				print(confusion_matrix(y_teste, predictions))
				print(classification_report(y_teste, predictions))
				print(accuracy_score(y_teste, predictions))
