import torch
import unittest

import sys
sys.path.append('.')

from bhtrace.geometry import SphericallySymmetric
from bhtrace.functional import sph2cart, cart2sph, 
from bhtrace.tracing import PTracer, CTracer

# WIP

class TestPrecision(unittest.TestCase):

    def setUp(self):
        pass

    def test_Horizon(self):

        ST = SphericallySymmetric()

        b_cr = torch.sqrt(27/4)

        X0 = torch.Tensor([[0.0, 100.0, b_cr, 0.0]])
        P0 = torch.Tensor([[1, -1, 0, 0]])

        gma0 = Photon(ST)
        P0 = gma0.GetNullMomentum(X0, P0)

        ctrace = CTracer()
        ptrace = PTracer()

        

    def test_ExactSchwarzschild(self):

        

        pass