# -*- coding: utf-8 -*-
from .labeling import Labeling
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

class MRA(Labeling):
	"""docstring for MRA"""
	def __init__(self, original_clusters, clusters, attributes, variacao, edges):
		super(MRA, self).__init__()
		self.original_clusters = original_clusters # clusters com valores continuos
		self.clusters = clusters # clusters com valores discretos
		self.attributes = [attr for attr in attributes]
		self.variacao = variacao
		self.edges = edges


	def execute(self):
		self.relevancies = {}

		# percorre cada cluster
		for cluster in self.clusters:
			relevance = {} # Armazena a relevância de cada atributo para o cluster atual

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

		self.most_rel = self.select_attributes()
		self.most_freq = self.calc_frequency()
		self.calc_label()
		self.calc_accuracy()

		# print("most_freq: valores mais frequentes dos atributos mais relevantes\n", self.most_freq)
		# print()
		# print("self.labels: rotulos dos clusters\n", self.labels)
		# print()
		# print("self.accuracy: quantidade de acertos de cada atributo\n", self.accuracy)

		self.make_report()
		print(self.report_)



	def select_attributes(self):
		"""Seleciona os atributos mais relevantes de acordo com o parametro de variacao"""
		most_rel = {}
		for cluster, relevance in self.relevancies.items():
			most_rel_attr = {} # atributos mais relevantes para o cluster atual
			maximo = max(relevance.values())
			for attr, rel in relevance.items():
				if rel >= maximo - self.variacao:
					most_rel_attr[attr] = rel
			most_rel[cluster] = most_rel_attr

		return most_rel



	def calc_frequency(self):
		"""Busca os valores mais frequentes dos atributos mais relevantes."""
		most_freq = {}
		for cluster, rel_attr in self.most_rel.items():
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

		return most_freq



	def calc_label(self):
		self.labels = {}

		for cluster, attrs in self.most_freq.items():
			label = {}
			for attr, track in attrs.items():
				edge = self.edges[self.attributes.index(attr)]
				label[attr] = (edge[track-1], edge[track])

			self.labels[cluster] = label


	def calc_accuracy(self):
		"""
		Calcula a quantidade de elementos que obedecem aos valores dos
		atributos para cada cluster
		"""
		self.accuracy = {}

		for cluster, attrs in self.labels.items():
			cluster_accuracy = {}
			acerto_cluster = 0

			for element in self.original_clusters[cluster]:
				correto = True

				for attr, interval in attrs.items():
					attr_index = self.attributes.index(attr)

					if interval[0] <= element[attr_index] <= interval[1]:
						if attr in cluster_accuracy.keys():
							cluster_accuracy[attr] += 1
						else:
							cluster_accuracy[attr] = 1
					else:
						correto = False

				if correto:
					acerto_cluster += 1

			cluster_accuracy['total'] = acerto_cluster

			self.accuracy[cluster] = cluster_accuracy


	def make_report(self):
		"""Monta o relatorio com o resultado."""
		clusters = list(self.labels.keys())
		clusters.sort()
		general_accuracy = 0 # quantidade de elementos que obedece a todos os intervalos de valores do rotulo
		n_elements = 0

		self.report = ""

		self.report += "Relevancia de todos os atributos:\n"
		for cluster in clusters:
			self.report += " Cluster: " + cluster + "\n"
			for attr, relevance in self.relevancies[cluster].items():
				self.report += ("   %s: %.2f%%\n" % (attr, relevance))

			self.report += "\n"

		self.report += ("Atributos mais relevantes (Variacao = %d):\n" % self.variacao)
		for cluster in clusters:
			self.report += " Cluster: " + cluster + "\n"
			for attr, relevance in self.most_rel[cluster].items():
				self.report += ("   %s: %.2f%%\n" % (attr, relevance))

			self.report += "\n"

		self.report += "Rotulos:\n"
		for cluster in clusters:
			self.report += " Cluster: " + cluster + "\n"

			for attr, interval in self.labels[cluster].items():
				accuracy = (self.accuracy[cluster][attr] / len(self.clusters[cluster])) * 100
				bracket = "[" if self.most_freq[cluster][attr] == 1 else "]"
				self.report += ("   %s: %s%.4f , %.4f] | %.2f%% (%d/%d)\n" % (attr,  bracket, interval[0], interval[1], accuracy, self.accuracy[cluster][attr], len(self.clusters[cluster])))

			acuracia_total = self.accuracy[cluster]['total'] / len(self.clusters[cluster]) * 100
			self.report += ("   Acerto total: %.2f%% (%d/%d)\n\n" % (acuracia_total, self.accuracy[cluster]['total'], len(self.clusters[cluster])))

			general_accuracy += self.accuracy[cluster]['total']
			n_elements += len(self.clusters[cluster])

		self.report += (" Acuracia geral: %.2f%% (%d/%d)\n" % ((general_accuracy/n_elements) * 100, general_accuracy, n_elements))


	@property
	def report_(self):
		"""Retorna o texto do ralatorio."""
		return self.report
