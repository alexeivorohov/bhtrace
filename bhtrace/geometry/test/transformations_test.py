import os
import sys
import inspect

import unittest
import torch

from bhtrace.geometry.transformation import CoordinateTransformation
from bhtrace.geometry.transformation_collection import (
    Ident,
    Cartesian2Spherical,
    Spherical2Cartesian,
    Cartesian2Axial,
    Axial2Cartesian,
    Shift,
    Rotation
)

_TRANSFORMATIONS_ = {
    'Ident': Ident,
    'Cart2Sph': Cartesian2Spherical,
    'Sph2Cart': Spherical2Cartesian,
    'Cart2Ax': Cartesian2Axial,
    'Ax2Cart': Axial2Cartesian,
    'Shift': Shift,
    'Rotation': Rotation
}


class MockTransformation(CoordinateTransformation):
    def __init__(self, inverse=None):
        super().__init__()
        if inverse is None:
            self.inverse = MockTransformation(inverse=self)
        else:
            self.inverse = inverse

    def forward(self, X, **kwargs):
        return X

    def jac(self, X, **kwargs):
        I = torch.eye(4, dtype=X.dtype, device=X.device)
        return I.repeat(*X.shape[:-1], 1, 1)

class TestTransformations(unittest.TestCase):

    def setUp(self):
        # Prepare a dictionary of transformation instances for testing
        self.TFS = {'mock': MockTransformation()}
        # Provide minimal init args for known transformations
        init_args = {
            'Shift': {'pos': torch.randn(4)},
            'Ax2Cart': {},
            'Cart2Ax': {},
            'Rotation': {'pos': torch.randn(4)} # Rotation needs pos
        }
        keys = init_args.keys() & _TRANSFORMATIONS_.keys()
        for key in keys:
            try:
                self.TFS[key] = _TRANSFORMATIONS_[key](**init_args[key])
            except Exception as e:
                print(f'Cannot instantiate transformation {key},\ninit_args: {init_args[key]}\nError: {e}')
        
        self.transformation_classes = list(_TRANSFORMATIONS_.values()) + [MockTransformation]
        self.atol = 1e-4
        self.rtol = 1e-4

    def test_initialization(self):
        '''
        Test that all CoordinateTransformation implementations can be instantiated.

        AI, Checked
        '''
        for cls in self.transformation_classes:
            # Try to instantiate with no args, or with dummy args if needed
            try:
                instance = cls()
            except TypeError:
                # Try with dummy args if possible
                sig = inspect.signature(cls)
                kwargs = {}
                for name, param in sig.parameters.items():
                    if param.default is not inspect.Parameter.empty:
                        continue
                    # Provide dummy values for common parameter names
                    if name == 'pos':
                        kwargs[name] = torch.randn(4)
                    elif name == 'inverse':
                        kwargs[name] = None
                    else:
                        pass
                try:
                    instance = cls(**kwargs)
                except Exception as e:
                    self.fail(f"Could not instantiate {cls.__name__} with dummy args: {e}")
            except Exception as e:
                self.fail(f"Could not instantiate {cls.__name__}: {e}")

            self.assertIsInstance(instance, CoordinateTransformation, f"{cls.__name__} is not a CoordinateTransformation")
            
        for key, TF in self.TFS.items():
            self.assertIsInstance(TF, CoordinateTransformation, f"{key} is not a CoordinateTransformation")
            self.assertIsNotNone(TF.inverse, f"{key} inverse is None")
            self.assertIsInstance(TF.inverse, CoordinateTransformation, f"{key} inverse is not a CoordinateTransformation")
            self.assertIs(TF.inverse.inverse, TF, f"{key} inverse's inverse is not itself")

    def test_inverse_consistency(self):
        '''
        Test that applying transformation and its inverse returns the original input
        '''
        for key, TF in self.TFS.items():
            if key in ['Ax2Cart', 'Cart2Ax', 'Rotation']:
                continue
            X = torch.randn(2, 4)
            try:
                X1 = TF(X)
                X2 = TF.inverse(X1)
                self.assertTrue(torch.allclose(X, X2, atol=self.atol, rtol=self.rtol), f"{key} inverse consistency failed")
            except NotImplementedError:
                pass


    def test_jac(self):
        '''
        Test jacobian and inverse jacobian
        '''
        for key, TF in self.TFS.items():
            if key in ['Ax2Cart', 'Cart2Ax', 'Rotation']:
                continue
            X_0 = torch.randn(2, 3, 4)*10
            
            try:
                X_1 = TF(X=X_0)
                J_0 = TF.jac(X_0)
                J_1 = TF.inverse.jac(X_1)
            except NotImplementedError:
                continue

            self.assertEqual(list(J_0.shape), [*X_0.shape, 4], f'Jacobian has incorrect shape: {J_0.shape}, TF: {key}')
            self.assertEqual(list(J_1.shape), [*X_1.shape, 4], f'Jacobian inv has incorrect shape: {J_1.shape}, TF: {key}')
            p = torch.einsum('...pq, ...qu -> ...pu', J_0, J_1)
            I = torch.eye(4).repeat(*X_0.shape[:-1], 1, 1)
            self.assertEqual(list(p.shape), [*X_0.shape, 4], f'J @ J_inv has incorrect shape: {p.shape}, TF: {key}')
            self.assertTrue(torch.allclose(p, I, atol=self.atol, rtol=self.rtol), f'J @ J_inv is not equal I for {key}')


    def test_tensor_valence(self):
        '''
        Test tensor transformation with valence argument
        '''
        for key, TF in self.TFS.items():
            if key in ['Ax2Cart', 'Cart2Ax', 'Rotation', 'Shift']:
                continue
            
            batch_shape = (10,)
            X = torch.randn(*batch_shape, 4) * 10

            test_configs = {
                "vector(default)": (None, torch.randn(*batch_shape, 4)),
                "vector": ([True], torch.randn(*batch_shape, 4)),
                "covector": ([False], torch.randn(*batch_shape, 4)),
                "(1,1)-tensor": ([True, False], torch.randn(*batch_shape, 4, 4)),
                "(0,2)-tensor": ([False, False], torch.randn(*batch_shape, 4, 4)),
            }

            for name, (valence, A) in test_configs.items():
                try:
                    X_new, A_new = TF.tensor(X, A, valence=valence)
                    self.assertEqual(X_new.shape, X.shape)
                    self.assertEqual(A_new.shape, A.shape)
                    _, A_back = TF.inverse.tensor(X_new, A_new, valence=valence)
                    self.assertTrue(torch.allclose(A, A_back, atol=self.atol, rtol=self.rtol),
                                    f"{key} {name} transformation consistency failed")
                except NotImplementedError:
                    pass

    
    # def test_JIT(self):
    #     '''
    #     Test if transformation is jit-compilable
    #     '''
    #     scripted_tf = torch.jit.script(self.TFS['mock'])
    #     self.assertTrue(callable(scripted_tf.__call__))
    #     self.assertTrue(callable(scripted_tf.jac))
    #     self.assertTrue(callable(scripted_tf.tensor))


if __name__ == '__main__':
    unittest.main()

