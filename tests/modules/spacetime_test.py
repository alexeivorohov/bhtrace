import torch
import unittest

import sys
sys.path.append('.')

from bhtrace.geometry import SphericallySymmetric
from bhtrace.functional import sph2cart, cart2sph, 
from bhtrace.tracing import PTracer, CTracer


class TestSpacetimeA(unittest.TestCase):
    
    def __init__(self):

        class mock_spacetime(Spacetime):

    
            def g(self, X):
                
                pass


            def ginv(self, X):

                pass
        
            
            def crit(self, X):

                pass

        self.mock_st = mock_spacetime()

    
    def test_metric(self):

        pass


    def test_derivatives(self):

        pass