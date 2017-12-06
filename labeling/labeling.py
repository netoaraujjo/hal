# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class Labeling(ABC):
    """docstring for Labeling."""
    def __init__(self):
        super(Labeling, self).__init__()


    @abstractmethod
    def execute(self):
        pass


    @abstractmethod
    def make_report(self):
        pass


    @property
    @abstractmethod
    def report_(self):
        pass
