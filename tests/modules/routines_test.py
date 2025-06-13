import unittest
import torch
import numpy as np

import sys
import os
root_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.functional.routines import cart2sph, sph2cart, points_generate, net, bisection, def_fspace, levi_civita_tensor, print_status_bar

class TestRoutines(unittest.TestCase):

    def test_cart2sph(self):
        inX = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        inP = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        outX, outP = cart2sph(inX, inP)
        self.assertEqual(outX.shape, inX.shape)
        self.assertEqual(outP.shape, inP.shape)

    def test_sph2cart(self):
        inX = torch.tensor([[0.0, 1.0, np.pi / 2, 0.0]])
        inP = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        outX, outP = sph2cart(inX, inP)
        self.assertEqual(outX.shape, inX.shape)
        self.assertEqual(outP.shape, inP.shape)

    def test_points_generate(self):
        ts = [0.0, 1.0]
        rs = [1.0, 2.0]
        ths = [0.0, np.pi / 2]
        phs = [0.0, np.pi]
        X = points_generate(ts, rs, ths, phs)
        self.assertEqual(X.shape, (len(ts) * len(rs) * len(ths) * len(phs), 4))

    def test_net(self):
        xx, yy, zz = net(shape='square', rng=(5, 5), YZ0=[0, 0], X0=20, YZsize=[8, 8])
        self.assertEqual(xx.shape, yy.shape)
        self.assertEqual(yy.shape, zz.shape)

    def test_bisection(self):
        func = lambda x, par: x**2 - par
        x_min = torch.tensor([-1.0])
        x_max = torch.tensor([2.0])
        par = torch.tensor([1.0])
        zero = bisection(func, x_min, x_max, par)
        self.assertTrue(torch.allclose(zero, torch.tensor([1.0]), atol=1e-4))

    # Should be fixed and tested with more complex functions
    def test_def_fspace(self):
        def func(x, par):
            return torch.where(torch.abs(x) < par, x, torch.tensor(float('nan')))

        par = torch.tensor([1.0])
        x_min = -2 * par
        x_max = 2 * par

        x_min, x_max = def_fspace(func, x_min, x_max, par)
        self.assertTrue(torch.allclose(x_max, par, atol=1e-4))
        self.assertTrue(torch.allclose(x_min, -par, atol=1e-4))

    def test_levi_civita_tensor(self):
        dim = 3
        tensor = levi_civita_tensor(dim)
        self.assertEqual(tensor.shape, (dim, dim, dim))

    def test_print_status_bar(self):
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        print_status_bar(5, 10, 1.0)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn('50.00%', output)

if __name__ == '__main__':
    unittest.main()