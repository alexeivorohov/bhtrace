import torch
import time
import unittest
import inspect

import sys
import os

from bhtrace.geometry.spacetime import MinkowskiCart, MinkowskiSph
import bhtrace.physics.electrodynamics as bhE
import bhtrace.utils.units as bhU


MODELS = [
    bhE.Maxwell(bhU.si),
]

FIELDS = [
    bhE.PointCharge(0.1)
]

COORDS = [
    bhE.PointCharge(0.1)
]

def test_output_shapes():
    
    def test_init(self):
        '''
        Test class initialization
        '''
        for key, obj in self.instance_to_test.items():
            self.assertIsInstance(obj, self.base_class, f"{key} is not a ")

    def test_field_calculation(self):

        regimes = ['B', 'E', 'EB']
        model_type = 'F'

        # TODO: Different models
        ED = ELECTRODYNAMICS_REGISTRY['Maxwell']()
        
        ST = MinkowskiSph()

        # TODO: Different field configurations
        E = lambda X: torch.Tensor([0, -torch.pow(X[...,1],-2), 0, 0])
        B = lambda X: torch.Tensor([0, torch.pow(X[...,1],-2), 0, 0])
        
        ED.attach_fields(E, B)

        X = torch.randn(1, 4)*10
        U = torch.tensor([1, 0, 0, 0]).repeat(*X.shape[:-1], 1).float()

        gX = ST.g(X).float()

        for regime in regimes:
        
            ED.set_regime(fields=regime, model_type=model_type)

            ED.calculate(X, gX, U)

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

    def test_save_load(self):
        '''
        Test if ED models can be saved and loaded.
        '''
        for name, ed in self.instance_to_test.items():
            with self.subTest(name=name):
                try:
                    state = ed.state()
                    new_ed = Electrodynamics.from_dict(state)
                    new_state = new_ed.state()
                    state.pop('name', None)
                    new_state.pop('name', None)
                    self.assertEqual(state, new_state)
                except Exception as e:
                    self.fail(f"Save/load failed for {name}: {e}")



if __name__ == '__main__':

    unittest.main()
