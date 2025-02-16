import torch
import unittest
from uniplot import plot as uplot

import sys
sys.path.append('.')

from bhtrace.functional import RKF23b, Euler



class TestODE(unittest.TestCase):

    def setUp(self):

        self.atol = 1e-2
        self.rtol = 1e-2
        self.eps = 1e-3

        self.solvers = [Euler(), RKF23b(varistep=True)]
        self.expected_tolerance = [1e-3, 1e-5]


    def test_Exponent(self):
        '''
        Test on simplest equation

        '''

        # parameters
        dim = 5
        nsteps = 128

        k = - torch.linspace(0, 1, dim)

        # equation
        def term(t, X):

            return k*X

        # test loop
        for ode in self.solvers:

            X0 = torch.ones(dim)
            T0 = torch.Tensor([0.0])
    
            sol = ode.forward(term, T0, X0, dt=0.01, nsteps=nsteps, eps=self.eps)

            exact = X0*torch.exp(torch.outer(sol['T'], k))

            crit1 = torch.allclose(sol['X'], exact, atol=self.atol, rtol=self.rtol)
            self.assertTrue(crit1, 'Tolerance test not passed')


    def test_OscillatorTask(self):

        # parameters
        k = 0.5
        mu = 0.0

        # ODE term
        def term(t, XP):

            dX = XP[1]
            dP = - k*XP[0] - mu*XP[1]

            return torch.Tensor([dX, dP])

        # hamiltonian ()
        def H(t, X, P):

            H = k*torch.pow(X, 2) + torch.pow(P, 2)

            return H

        # test loop
        for ode in self.solvers:
            XP0 = torch.Tensor([1.0, 0])
            T0 = torch.Tensor([0.0])

            sol = ode.forward(term, T0, XP0, dt=0.01, nsteps=16, eps=self.eps)

            X = sol['X'][:, 0]
            P = sol['X'][:, 1]
            T = sol['T']
            LTE = sol['LTE']

            uplot(xs=X, ys=P, lines=True, title='Phase portrait')
            uplot(xs=T, ys=H(T, X, P), lines=True, title='Energy vs T')
            uplot(xs=[T, T], ys=[LTE[:, 0], LTE[:, 1]], \
                lines=True, title='LTE vs T', color=True,\
                legend_labels = ['X', 'P'])

            crit1 = torch.all(torch.less(sol['LTE'], 1e-4))
            crit2 = torch.all(torch.less(sol['LTE'], 1e-6))
            crit3 = torch.all(torch.less(sol['LTE'], 1e-8))
            print(T)

            self.assertTrue(crit1, 'Minimal tolerance test not passed')



if __name__ == '__main__':
    unittest.main()




        


        
        
        