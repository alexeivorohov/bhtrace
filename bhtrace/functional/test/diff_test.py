import unittest
import torch

import sys
import os

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
sys.path.append(os.getcwd())


from bhtrace.functional.diff import Grad, D

class TestDiff(unittest.TestCase):

    def setUp(self):
        
        self.f_dict = {
            'const': lambda X: torch.ones_like(X),
            'linear': lambda X: 2*X,
            'exp': lambda X: torch.exp(X)
        }

        self.expected_df = {
            'const': lambda X: torch.zeros_like(X),
            'linear': lambda X: 2*torch.ones_like(X),
            'exp': lambda X: torch.exp(X)
        }

        self.func_param = {
            'wx2' : lambda X, params: torch.pow(X, 2)*params
        }

        self.tolerances = [
            1e-2,
            1e-4,
            ]


    def test_diff_linear(self):
        '''
        Test derivative calculation
        '''
        
        dim = 3
        X = torch.randn(1, 2, dim)*5
        X.to(dtype=torch.float64)

        for key, f in self.f_dict.items():
            for eps in self.tolerances:

                df = D(f, eps=eps, order=1)
                result = df(X)
                expected = self.expected_df[key](X)

                e = torch.abs(result - expected).mean()
                
                message = f'Given tolerance {eps} is not acheived: MAE= {e}\n' + \
                    f'Result:\n\t{result}\nExpected:\n\t{expected}\n' + \
                    f'Function: {key}, scheme order: linear \n'
                
                tol_criterion = torch.allclose(result, expected, atol=eps*10, rtol=eps*10)

                self.assertTrue(tol_criterion, message)


    def test_diff_horder(self):

        dim = 3
        X = torch.randn(1, 2, dim)*5
        X.to(dtype=torch.float64)
        orders = [2, 4]

        for order in orders:
            for key, f in self.f_dict.items():
                for eps in self.tolerances:

                    df = D(f, eps=eps, order=order)
                    result = df(X)
                    expected = self.expected_df[key](X)

                    e = torch.abs(result - expected).mean()
                    
                    tol_condition = torch.allclose(result, expected, atol=eps*10, rtol=eps*10)

                    message = f'Given tolerance {eps} is not acheived: MAE= {e}\n' + \
                        f'Result:\n\t{result}\nExpected:\n\t{expected}\n' + \
                        f'Function: {key}, scheme order: {order} \n'

                    self.assertTrue(tol_condition, message)
                                    

    def test_grad_linear(self):

        dim = 4
        X = torch.randn(1, 2, dim)*5
        X.to(dtype=torch.float64)

        f_dict = {
            'const': lambda X: torch.ones(*X.shape[:-1]),
            'linear': lambda X: torch.zeros()
        }

        for key, f in f_dict.items():
            for eps in self.tolerances:

                df = D(f, eps=eps, order=1)
                result = df(X)
                expected = self.expected_df[key](X)

                e = torch.abs(result - expected).mean()
                
                message = f'Given tolerance {eps} is not acheived: MAE= {e}\n' + \
                    f'Result:\n\t{result}\nExpected:\n\t{expected}\n' + \
                    f'Function: {key}, scheme order: linear \n'
                
                tol_criterion = torch.allclose(result, expected, atol=eps*10, rtol=eps*10)

                self.assertTrue(tol_criterion, message)

        


    # TODO: repair this test
    # def test_hessian(self):
    #     hessian = Hessian(self.func)
    #     result = hessian(self.X, self.params)
    #     expected = torch.diag(torch.tensor([2.0, 4.0, 6.0, 8.0]))
    #     self.assertTrue(torch.allclose(result, expected, atol=1e-4))


    # def test_diff_linear(self):
    #     diff_linear = DiffLinear(self.func)
    #     result = diff_linear(self.X, self.params)
    #     expected = torch.tensor([2.0, 8.0, 18.0, 32.0])
    #     self.assertTrue(torch.allclose(result, expected, atol=1e-4))


    # def test_diff_horder_order_2(self):
    #     diff_horder = DiffHOrder(self.func, order=2)
    #     result = diff_horder(self.X, self.params)
    #     expected = torch.tensor([2.0, 8.0, 18.0, 32.0])
    #     self.assertTrue(torch.allclose(result, expected, atol=1e-4))


    # def test_diff_horder_order_4(self):
    #     diff_horder = DiffHOrder(self.func, order=4)
    #     result = diff_horder(self.X, self.params)
    #     expected = torch.tensor([2.0, 8.0, 18.0, 32.0])
    #     self.assertTrue(torch.allclose(result, expected, atol=1e-4))


if __name__ == '__main__':

    unittest.main()