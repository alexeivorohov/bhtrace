import os
import sys

root_path = '/home/alexey/Work/bhtrace-dev'
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.geometry.transformations import CoordinateTransformation
from bhtrace.geometry import _TRANSFORMATIONS_

import unittest
import torch


class MockTransformation(CoordinateTransformation):
    '''
    Transformation class, useful for testing;

    Performs an ident transformation.
    '''

    def __init__(self, inverse=None):

        if inverse == None:
            super().__init__(inverse=self)
        else:
            super().__init__(inverse=inverse)
        
        pass


    def __call__(self, X: torch.Tensor):

        return X
    

    def jac(self, X: torch.Tensor):

        return torch.eye(4).repeat(*X.shape[:-1], 1, 1)
    

class TestTransformations(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):

        super(TestTransformations, self).__init__(*args, **kwargs)

        self.TFS = {'mock': MockTransformation()}

        init_args = {
            'Shift': {'pos': torch.randn(4)},
            'Ax2Cart': {},
            'Cart2Ax': {}
        }

        keys = init_args.keys() & _TRANSFORMATIONS_.keys()
        
        for key in keys:
            try:
                self.TFS[key] = _TRANSFORMATIONS_[key](**init_args[key])
            except:
                print(f'Can not instanate transformation {key},\ninit_args: {init_args[key]}')
        
        print(self.TFS.keys())

        self.atol = 1e-4
        self.rtol = 1e-4

    
    def test_init(self):
        '''
        Test class initialization
        '''
        for key, TF in self.TFS.items():
            self.assertIsInstance(TF, CoordinateTransformation, f"{key} is not a CoordinateTransformation")
            self.assertIsNotNone(TF.inverse, f"{key} inverse is None")
            self.assertIsInstance(TF.inverse, CoordinateTransformation, f"{key} inverse is not a CoordinateTransformation")
            # Inverse of inverse should be the original
            self.assertIs(TF.inverse.inverse, TF, f"{key} inverse's inverse is not itself")


    def test_shift(self):
        '''
        Test Shift transformation
        '''
        if 'Shift' in self.TFS:
            TF = self.TFS['Shift']
            X = torch.randn(5, 4)
            pos = TF.pos
            X_shifted = TF(X)
            self.assertTrue(torch.allclose(X_shifted, X + pos))
            X_back = TF.inverse(X_shifted)
            self.assertTrue(torch.allclose(X_back, X, atol=self.atol, rtol=self.rtol))


    def test_cart2ax_ax2cart(self):
        '''
        Test Cart2Ax and Ax2Cart transformations
        '''
        if 'Cart2Ax' in self.TFS and 'Ax2Cart' in self.TFS:
            cart2ax = self.TFS['Cart2Ax']
            ax2cart = self.TFS['Ax2Cart']
            X = torch.randn(7, 4)
            X_ax = cart2ax(X)
            X_cart = ax2cart(X_ax)
            self.assertEqual(X.shape, X_cart.shape)
            # Allow for some numerical error due to sqrt/arctan
            self.assertTrue(torch.allclose(X, X_cart, atol=1e-4, rtol=1e-4))


    def test_mock_tensor(self):
        '''
        Test tensor transformation for MockTransformation (identity)
        '''
        TF = self.TFS['mock']
        X = torch.randn(3, 4)
        A = torch.randn(3, 4, 4)
        X_new, A_new = TF.tensor(X, A)
        self.assertTrue(torch.allclose(X, X_new))
        self.assertTrue(torch.allclose(A, A_new))


    def test_tensor_no_valence(self):
        '''
        Test tensor transformation with no valence argument
        '''
        for key, TF in self.TFS.items():
            X = torch.randn(2, 4)
            A = torch.randn(2, 4)
            try:
                X_new, A_new = TF.tensor(X, A)
                self.assertEqual(X_new.shape, X.shape)
                self.assertEqual(A_new.shape, A.shape)
            except NotImplementedError:
                pass


    def test_tensor_valence(self):
        '''
        Test tensor transformation with valence argument
        '''
        for key, TF in self.TFS.items():
            X = torch.randn(2, 4)
            A = torch.randn(2, 4, 4)
            valence = [True, False]
            try:
                X_new, A_new = TF.tensor(X, A, valence=valence)
                self.assertEqual(X_new.shape, X.shape)
                self.assertEqual(A_new.shape, A.shape)
            except NotImplementedError:
                pass


    def test_jac_shape_and_eye(self):
        '''
        Test that jacobian is identity for MockTransformation and Shift
        '''
        for key in ['mock', 'Shift']:
            if key in self.TFS:
                TF = self.TFS[key]
                X = torch.randn(2, 3, 4)
                J = TF.jac(X)
                I = torch.eye(4).repeat(*X.shape[:-1], 1, 1)
                self.assertTrue(torch.allclose(J, I, atol=1e-5), f"{key} jacobian is not identity")


    def test_inverse_consistency(self):
        '''
        Test that applying transformation and its inverse returns the original input
        '''
        for key, TF in self.TFS.items():
            X = torch.randn(2, 4)
            try:
                X1 = TF(X)
                X2 = TF.inverse(X1)
                self.assertTrue(torch.allclose(X, X2, atol=1e-4), f"{key} inverse consistency failed")
            except NotImplementedError:
                pass

    def test_repr_str(self):
        '''
        Test __repr__ and __str__ if implemented
        '''
        for key, TF in self.TFS.items():
            self.assertTrue(isinstance(str(TF), str))
            self.assertTrue(isinstance(repr(TF), str))


    def test_call(self):
        '''
        Test forward and inverse calls
        '''

        X_0 = torch.randn(2, 10, 4)

        for key, TF in self.TFS.items():


            X_1 = TF(X = X_0)
            X_2 = TF.inverse(X = X_1)

            self.assertTrue(torch.allclose(X_2, X_0, atol=1e-5),
                            f'Transformation: {key}')


    def test_jac(self):
        '''
        Test jacobian and inverse jacobian
        '''
        
        for key, TF in self.TFS.items():

            X_0 = torch.randn(2, 3, 4)

            X_1 = TF(X = X_0)
            
            J_0 = TF.jac(X_0)
            J_1 = TF.inverse.jac(X_1)

            print(type(J_0))
            print(type(J_1))

            self.assertTrue((list(J_0.shape) == [*X_0.shape, 4]),\
                            f'Jacobian has incorrect shape: {J_0.shape}, TF: {key}'
                            )
            
            self.assertTrue((list(J_1.shape) == [*X_0.shape, 4]),\
                            f'Jacobian inv has incorrect shape: {J_1.shape}, TF: {key}'
                            )

            p = torch.einsum('mnpq, mnqu -> mnpu', J_0, J_1)

            I = torch.eye(4).repeat(*X_0.shape[:-1],1,1)

            self.assertTrue((list(p.shape) == [*X_0.shape, 4]),\
                            f'J @ J_inv has incorrect shape: {p.shape}, TF: {key}'
                            )

            self.assertTrue(torch.allclose(p, I, atol=1e-4),
                            f'J @ J_inv is not equal I, TF: {key}'
                            )


    def test_tensor_no_valence(self):


        pass

    
    def test_tensor_valence(self):

        pass


    # def test_JIT(self):
    #     '''
    #     Test if transformation is jit-compilable
    #     '''
    #     scripted_tf = torch.jit.script(self.mock)

    #     self.assertTrue(callable(scripted_tf.__call__))
    #     self.assertTrue(callable(scripted_tf.jac))
    #     self.assertTrue(callable(scripted_tf.tensor))


if __name__ == '__main__':

    unittest.main()

