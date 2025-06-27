import torch
import time
import unittest
import inspect

import sys
import os

root_path = '/home/alexey/Work/bhtrace-dev'
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.geometry import MinkowskiCart, MinkowskiSph
from bhtrace.electrodynamics import Electrodynamics, _ED_MODELS_


class TestElectrodynamics(unittest.TestCase):
    
    
    def __init__(self, *args, **kwargs):
        '''
        Collect all subclasses of Electrodynamics and instantate tests
        '''
        super(TestElectrodynamics, self).__init__(*args, **kwargs)

        self.base_class = Electrodynamics

        self.impls = {}

        all_impls = inspect.getmembers(
            sys.modules['bhtrace.electrodynamics'], inspect.isclass
        )

        for name, obj in all_impls:
            if issubclass(obj, self.base_class) and obj is not self.base_class:
                self.impls[name] = obj
        
        keys_all = self.impls.keys()
        print(f'All impls: {keys_all}')

        keys_cross = (_ED_MODELS_.keys() & 
                      self.impls.keys())
        print(f'Supported impls: {keys_cross}')

        self.instance_to_test = {}
        for key in keys_all:
            try:
                self.instance_to_test[key] = self.impls[key]()
            except Exception as e:
                print(f'Cannot instantiate ED model {key}')
    
                
        self.atol = 1e-4
        self.rtol = 1e-4

    
    def test_init(self):
        '''
        Test class initialization
        '''
        for key, obj in self.instance_to_test.items():
            self.assertIsInstance(obj, self.base_class, f"{key} is not a ")
            # self.assertIsNotNone(TF.inverse, f"{key} inverse is None")
            # self.assertIsInstance(TF.inverse, CoordinateTransformation, f"{key} inverse is not a CoordinateTransformation")
            # self.assertIs(TF.inverse.inverse, TF, f"{key} inverse's inverse is not itself")


    def test_field_calculation(self):

        regimes = ['B', 'E', 'EB']
        model_type = 'F'

        # TODO: Different models
        ED = _ED_MODELS_['Maxwell']()
        
        ST = MinkowskiSph()

        # TODO: Different field configurations
        E = lambda X: torch.Tensor([0, -torch.pow(X[...,1],-2), 0, 0])
        B = lambda X: torch.Tensor([0, torch.pow(X[...,1],-2), 0, 0])
        
        ED.attach_fields(E, B)

        X = torch.randn(1, 4)*10

        gX = ST.g(X)

        for regime in regimes:
        
            ED.set_regime(fields=regime, model_type=model_type)

            ED.calculate(X, gX)

            ## Check Maxwell tensor properties:
            # Shape:
            self.assertTrue(
                [*ED._Fuv.shape] == [*X.shape, 4],
                'Incorrect shape of Maxwell tensor' +
                f'\n get {[*ED._Fuv.shape]}, ' +
                f'expected {[*X.shape, 4]}'
            )
            # Symmetries:
            self.assertTrue(
                torch.allclose(
                    - ED._Fuv, 
                    torch.swapaxes(ED._Fuv, -1, -2),
                    atol=self.atol, rtol=self.rtol),
                'Antisymmery of Maxwell tensor does not hold'
            )

            # Check that two methods of F invariant computation lead to the same result:

            I1 = torch.einsum(
                '...pq, ...uv,...pu,...qv -> ...',
                ED._Fuv, ED._Fuv, gX, gX
            )

            idx0 = [0 for _ in I1.shape]
            self.assertTrue(
                torch.allclose(ED._F, I1, atol=self.atol, rtol=self.rtol),
                'Two computations of F invariant do not coincide\n' +
                f'{idx0}:(model._F: {ED._F[idx0]}) != (I1: {I1[idx0]} ) '
            )


    def test_model_calculation(self):

        regime = 'EB'
        model_types = ['F', 'FG']

        ED = _ED_MODELS_['Maxwell']()
        
        ST = MinkowskiSph()

        # TODO: Different field configurations
        E = lambda X: torch.Tensor([0, -torch.pow(X[...,1],-2), 0, 0])
        B = lambda X: torch.Tensor([0, torch.pow(X[...,1],-2), 0, 0])
        
        ED.attach_fields(E, B)

        X = torch.randn(1, 4)*10

        gX = ST.g(X)

        for model_type in model_types:

            ED.set_regime(fields=regime, model_type=model_type)

            ED.calculate(X, gX)



if __name__ == '__main__':

    unittest.main()
