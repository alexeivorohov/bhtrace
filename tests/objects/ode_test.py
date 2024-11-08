import torch
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

from bhtrace.functional import RKF23b


class TestODE_RKF23b(unittest.TestCase):

    def setUp(self):

        self.atol = 1e-6
        self.rtol = 1e-6
        self.eps = 1e-4

    def test_OscillatorTask(self):

        k = 1
        mu = 0.0
        ode = RKF23b()

        def term(t, XP):

            dX = XP[1]
            dP = - k*XP[0] - mu*XP[1]

            return torch.Tensor([dX, dP])

        XP0 = torch.Tensor([1.0, -0.5])
        T0 = torch.Tensor([0.0])

        sol = ode.forward(term, T0, XP0)

        X = sol['X'][0]
        P = sol['X'][1]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot(X, P)
        ax.grid('on')

        plt.show()

        crit1 = torch.all(torch.less(sol['err']), 1e-4)
        crit2 = torch.all(torch.less(sol['err']), 1e-6)
        crit3 = torch.all(torch.less(sol['err']), 1e-8)

        self.assertTrue(crit1, 'Minimal tolerance test not passed')
        print(crit1)
        print(crit2)
        print(crit3)


if __name__ == '__main__':
    unittest.main()




        


        
        
        