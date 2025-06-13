import torch
import time
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from bhtrace.geometry import EffGeomSPH, MinkowskiSph, Photon
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


    def test_PointReductionMaxw(self):
        
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
     
    def test_TraceReductionMaxw(self):

        # Constants and definitions
        q = 0.5
        q2 = q**2
        h = 10
        # a = 16*mu0*h
        a = 16*4*torch.pi*h
        a_10 = a/10

        # Models
        ED_dict = {
            'Maxwell': Maxwell(),
            'EulerHeisenberg_e': EulerHeisenberg(h=h)
            }

        # Metric functions
        f_dict = {
            'Maxwell': lambda r: 1.0 - 2.0*torch.pow(r, -1) + q2*torch.pow(r, -2), 
            'EulerHeisenberg_e': lambda r: 1.0 - 2.0*torch.pow(r, -1) + q2*torch.pow(r, -2),
            }

        df_dict = {
            'Maxwell': lambda r: 2.0*torch.pow(r, -2) - 2*q2*torch.pow(r, -3),
            'EulerHeisenberg_e': lambda r: 2.0*torch.pow(r, -2) - 2*q2*torch.pow(r, -3),
            }

        # Field of a point charge
        Er_dict = {
            'Maxwell': lambda r: q*torch.pow(r, -2),
            'EH': lambda r: q*torch.pow(r, -2) - a*q2*torch.pow(r, -6)
        }

        # fields
        B_dict = {
            'Maxwell': lambda X: torch.Tensor([0.0, 0.0, 0.0, 0.0]),
            'EulerHeisenberg_e': lambda X: torch.Tensor([0.0, 0.0, 0.0, 0.0]),
            }

        E_dict = {
            'Maxwell': lambda X: torch.Tensor([0.0, Er_dict['Maxwell'](X[1]), 0.0, 0.0]),
            'EulerHeisenberg_e': lambda X: torch.Tensor([0.0, Er_dict['EH'](X[1]), 0.0, 0.0]),        
            }


        # Initializing spacetimes and photons: #    

        ST_dict = {}

        for k in ED_dict.keys():

            ST_dict[k] = EffGeomSPH(
                ED=ED_dict[k],
                E=E_dict[k],
                B=B_dict[k],
                f=f_dict[k],
                f_r=df_dict[k]
                )


        # Attaching particles:

        P_dict = {}

        for k in ED_dict.keys():

            P_dict[k] = Photon(ST_dict[k])

        # Initial conditions                  #

        N_PHOTONS = 20
        b = 10
        X0, Y0, Z0 = net('line', rng=(N_PHOTONS, 0), X0=20.0, YZsize=[b, 0], YZ0=[b/2, 0])

        Ni = X0.shape[0]

        X0 = torch.stack([torch.zeros(Ni), X0, Y0, Z0], dim=1)
        P0 = torch.zeros(Ni, 4)
        P0[:, 0] = torch.ones(Ni)
        P0[:, 1] = -torch.ones(Ni)

        X0sph, P0sph = cart2sph(X0, P0)


        #######################################
        # Perform tracing and save:           #
        #######################################

        tracer = PTracer()

        SESSION_NAME = 'ReductionUnittestE'
        # lst = ['Maxwell', 'EulerHeisenberg_m', 'EulerHeisenberg_e', 'EulerHeisenberg_me']
        lst = ['EulerHeisenberg_e']

        for k in lst:

            P0sph_cov = torch.zeros(Ni, 4)
            for i in range(Ni):
                P0sph_cov[i, :] = P_dict[k].GetNullMomentum(X0sph[i, :], P0sph[i, 1:])

            tracer.forward(P_dict[k], X0sph, P0sph_cov, T=10.0, nsteps=128)
            tracer.save(
                '{}_{}_{}.pkl'.format(SESSION_NAME, Ni, k),
                directory='test_dir/')



if __name__ == '__main__':
    unittest.main()








