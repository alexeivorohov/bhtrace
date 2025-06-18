import unittest
import torch

import sys
import os

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
sys.path.append(os.getcwd())


from bhtrace.functional.diff import Grad, Hessian, DiffLinear, DiffHOrder

class TestDiff(unittest.TestCase):

    def setUp(self):
        # Define a simple quadratic function for testing
        self.func = lambda X, params: params[0] * X[0]**2 + params[1] * X[1]**2 + params[2] * X[2]**2 + params[3] * X[3]**2
        self.params = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.X = torch.tensor([1.0, 2.0, 3.0, 4.0])


    def test_grad(self):
        grad = Grad(self.func)
        result = grad(self.X, self.params)
        expected = torch.tensor([2.0, 8.0, 18.0, 32.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


    def test_hessian(self):
        hessian = Hessian(self.func)
        result = hessian(self.X, self.params)
        expected = torch.diag(torch.tensor([2.0, 4.0, 6.0, 8.0]))
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


    def test_diff_linear(self):
        diff_linear = DiffLinear(self.func)
        result = diff_linear(self.X, self.params)
        expected = torch.tensor([2.0, 8.0, 18.0, 32.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


    def test_diff_horder_order_2(self):
        diff_horder = DiffHOrder(self.func, order=2)
        result = diff_horder(self.X, self.params)
        expected = torch.tensor([2.0, 8.0, 18.0, 32.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


    def test_diff_horder_order_4(self):
        diff_horder = DiffHOrder(self.func, order=4)
        result = diff_horder(self.X, self.params)
        expected = torch.tensor([2.0, 8.0, 18.0, 32.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


if __name__ == '__main__':



    unittest.main()