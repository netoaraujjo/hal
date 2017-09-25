# -*- coding: utf-8 -*-

import os.path

class IOManager(object):
	"""IOManager. Classe responsável pela leitura e escrita de arquivos"""
	def __init__(self, file_path):
		super(IOManager, self).__init__()
		self.file_path = file_path
	

	@staticmethod
	def read_file(file_path):
		"""Lê o aquivo apontado pelo parâmetro file_path"""
		pass

	@staticmethod
	def write_file(content, file_name):
		"""
		Escreve o conteúdo do parâmetro content no arquivo definido pelo
		parâmetro file_name
		"""
		pass