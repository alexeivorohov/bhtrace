'''
This file describes procedures of numerical differentiation, used by the lib

'''
import torch
from abc import ABC, abstractmethod


class Diff(ABC):

    def __init__(self):

        pass

    @abstractmethod
    def __call__(self, X):

        pass


class Grad(ABC):

    def __init__(self):
        

        pass


    def __call__(self, *args, **kwds):


        pass


class Diff_linear(Diff):

    def __init__(self, func: callable, history=False):

        self.last_call =
        pass


    def __call__(self, X):
        
        
        pass


class Diff_horder(Diff):

    def __init__(self, func: callable):

        pass


    def __call__(self, X):


