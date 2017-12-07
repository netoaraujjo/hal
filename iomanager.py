# -*- coding: utf-8 -*-

import os.path
import sys
import csv

class IOManager(object):
	"""IOManager. Classe responsável pela leitura e escrita de arquivos"""
	def __init__(self):
		super(IOManager, self).__init__()


	@staticmethod
	def read_file(file_path, header = False):
		"""
		Lê o aquivo apontado pelo parâmetro file_path.
		O parâmetro header indica a existência de cabeçalho.
		"""
		try:
			with open(file_path) as csvfile:
				reader = csv.reader(csvfile)
				# input = [row for row in reader]
				content = list(reader)
				if header:
					attributes = content[0]
					data = []
					for row in content[1:]:
						data.append([float(x) for x in row])
					content = {'attributes': attributes, 'data': data}
				return content
		except Exception as e:
			print('Não foi possível ler o arquivo!')
			sys.exit(1)


	@staticmethod
	def write_file(content, file_name):
		"""
		Escreve o conteúdo do parâmetro content no arquivo definido pelo
		parâmetro file_name
		"""
		pass
