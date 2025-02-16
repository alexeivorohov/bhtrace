import torch
import time
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from bhtrace.geometry import EffGeomSPH, MinkowskiSph
from bhtrace.electrodynamics import Maxwell, EulerHeisenberg
from bhtrace.tracing import PTracer
from bhtrace.functional import cart2sph, sph2cart



# Setting up

class TestEffGeomSph(unittest.TestCase):

    def setUp(self):
        
        self.atol = 1e-6
        self.rtol = 1e-6
        pass


    def test_Reduction(self):

        
        E = lambda X: torch.zeros(4)
        B = lambda X: torch.zeros(4)


        ED0 = Maxwell()
        ED1 = EulerHeisenberg(h=0.0)

        ST0 = MinkowskiSph()
        Eff_ST = {
            'Maxwell':EffGeomSPH(ED=ED0, E=E, B=B), 
            'EulerHeisenberg0': EffGeomSPH(ED=ED1, E=E, B=B)
            }

        n_x = 10
        X_s = [torch.Tensor([i, 1+i, 1.57, i]) for i in range(n_x)]

        for name, ST in Eff_ST.items():
        
            for X in X_s:

                gX0 = ST0.g(X)
                ginvX0 = ST0.ginv(X)
                ginvX1 = ST.ginv(X)
                gX1 = ST.g(X)

                self.assertTrue(torch.allclose(ginvX0, ginvX1, atol=self.atol, rtol=self.rtol),\
                    'ginv(X) at X={} not  match. Spacetime {} not reduces to Minkowski in zero limit\n{}'.format(X, name, ginvX1))

            
                self.assertTrue(torch.allclose(gX0, gX1, atol=self.atol, rtol=self.rtol),\
                    'g(X) at X={} not  match. Spacetime {} not reduces to Minkowski in zero limit\n{}'.format(X, name, gX1))

    def test_Reduction_Maxw(self):
        
        q = 0.5
        E = lambda X: torch.Tensor([0, q/X[1], 0, 0])
        B = lambda X: torch.zeros(4)


        ED0 = Maxwell()
        ED1 = EulerHeisenberg(h=0.0)

        ST0 = EffGeomSPH(ED=ED0, E=E, B=B)
        Eff_ST = {
            'EulerHeisenberg0': EffGeomSPH(ED=ED1, E=E, B=B)
            }

        n_x = 10
        X_s = [torch.Tensor([i, 1+i, 1.57, i]) for i in range(n_x)]

        for name, ST in Eff_ST.items():
        
            for X in X_s:

                gX0 = ST0.g(X)
                ginvX0 = ST0.ginv(X)
                ginvX1 = ST.ginv(X)
                gX1 = ST.g(X)

                self.assertTrue(torch.allclose(ginvX0, ginvX1, atol=self.atol, rtol=self.rtol),\
                    'ginv(X) at X={} not  match. Spacetime {} not reduces to Maxwell in zero limit\n{}'.format(X, name, ginvX1))

            
                self.assertTrue(torch.allclose(gX0, gX1, atol=self.atol, rtol=self.rtol),\
                    'g(X) at X={} not  match. Spacetime {} not reduces to Maxwell in zero limit\n{}'.format(X, name, gX1))
     

if __name__ == '__main__':
    unittest.main()








