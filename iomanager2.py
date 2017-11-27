# -*- coding: utf-8 -*-

import pandas as pd

class IOManager2(object):
	"""docstring for IOManager2"""
	def __init__(self):
		super(IOManager2, self).__init__()


	def read_file(self, file_path):
		df = pd.read_csv(file_path)
		return df
