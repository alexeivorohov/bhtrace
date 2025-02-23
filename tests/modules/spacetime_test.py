import torch
import unittest

import sys
sys.path.append('.')

from bhtrace.geometry import mock_spacetime
from bhtrace.geometry import collection
from bhtrace.functional import sph2cart, cart2sph
from bhtrace.tracing import PTracer, CTracer


class TestSpacetimeBase(unittest.TestCase):
    
    def __init__(self):

        self.n_eval = 10
        self.mock_st = mock_spacetime()

    
    def test_metric(self):


        pass


    def test_1derivatives(self):

        
        pass


    def test_JIT(self):

        pass


class TestSpacetimeCollection(unittest.TestCase):

    def __init__(self):

        self.test_st_dict = {}


    def test_init(self):

        for spacetime in collection:

            self.test_st_dict[spacetime] = spacetime()

        pass


    def test_metric(self):

        pass


    def test_reduction(self):

        pass


    def test_JIT(self):
        '''
        Test if classes are jit-compilable:
        '''

        pass

