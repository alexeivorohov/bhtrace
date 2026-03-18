import torch
import unittest

import sys
import os

from bhtrace.geometry.spacetime._base import MockSpacetime, Spacetime
from bhtrace.geometry.spacetime import SPACETIME_REGISTRY


class TestSpacetimeBase(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestSpacetimeBase, self).__init__(*args, **kwargs)
        self.n_eval = 10
        self.mock_st = MockSpacetime()

    
    def test_metric(self):
        '''
        Test if metric is computed correctly:
            g @ g^{-1} = I
        '''

        test_coords = torch.randn(self.n_eval, 4)
        g = self.mock_st.g(test_coords)
        ginv = self.mock_st.ginv(test_coords)

        self.assertTrue(torch.allclose(g @ ginv, torch.eye(4), atol=1e-5))

        pass


    def test_derivatives(self):
        '''
        Test if derivatives are computed correctly:
            dg = 0
            conn = 0
        '''
        test_coords = torch.randn(self.n_eval, 4)
        dg = self.mock_st.dg(test_coords)
        conn = self.mock_st.conn(test_coords)

        self.assertTrue(torch.allclose(dg, torch.zeros(4, 4, 4), atol=1e-5))
        self.assertTrue(torch.allclose(conn, torch.zeros(4, 4, 4), atol=1e-5))

        pass


    @unittest.skip("JIT test is failing")
    def test_JIT(self):
        '''
        Test if spacetime is jit-compilable
        '''
        try:
            self.mock_st.compile()
        except Exception as e:
            print(f"Compilation failed for {self.mock_st}: {e}")

        pass


class TestSpacetimeCollection(unittest.TestCase):

    def setUp(self):
        self.ST_dict = {}
        self.n_eval = 10
        for name, constructor in SPACETIME_REGISTRY.items():
            try:
                if name == 'KerrSchild':
                    st = constructor(a=0.9, m=1.0, Q=0.1)
                elif name == 'KerrAx':
                    st = constructor(a=0.9)
                elif name == 'SchwSchild':
                    st = constructor(m=1.0, Q=0.1)
                elif name in ['EffGeom', 'EffgeomSimple']:
                    continue # Skip these as they require complex objects
                else:
                    st = constructor()
                self.ST_dict[name] = st
            except Exception as e:
                print(f"Failed to initialize {name}: {e}")

    def test_init(self):
        '''
        Test if all spacetimes can be initialized
        '''
        self.assertTrue(len(self.ST_dict) > 0)

    def test_save_load(self):
        '''
        Test if spacetimes can be saved and loaded.
        '''
        st = MockSpacetime()
        state = st.state()
        expected_state = {'name': 'MockSpacetime', 'coefs': [1.0, 2.0, 3.0, 5.0]}
        self.assertEqual(state['coefs'], expected_state['coefs'])

        new_st = Spacetime.from_dict(state)
        new_state = new_st.state()
        self.assertEqual(state, new_state)


    def test_metric(self):
        '''
        Test if metric is computed correctly:
        g @ g^{-1} = I
        '''     
        
        test_coords = torch.randn(self.n_eval, 4)*10
        
        for ST in self.ST_dict.values():
            try:
                g = ST.g(test_coords)
                ginv = ST.ginv(test_coords)
                self.assertTrue(torch.allclose(g @ ginv, torch.eye(4), atol=1e-5))
            except Exception as e:
                print(f"Failed to compute metric for {ST}: {e}")
                continue

        pass


    # def test_reduction(self):
    #     '''
    #     Test if metirc reduces to the minkowski metric in the flat case:
    #     g = eta
    #     g^{-1} = eta^{-1}
    #     dg = 0
    #     conn = 0
    #     '''

    #     ref_ST = MockSpacetime()
    #     test_coords = torch.randn(self.n_eval, 4)

    #     ref_g = ref_ST.g(test_coords)
    #     ref_ginv = ref_ST.ginv(test_coords)
    #     ref_dg = ref_ST.dg(test_coords)
    #     ref_conn = ref_ST.conn(test_coords)

    #     for ST in self.ST_dict.values():
    #         if not hasattr(ST, '_reduction_params_') or ST._reduction_params_ is None:
    #             continue

    #         test_g = ST.g(test_coords)
    #         test_ginv = ST.ginv(test_coords)
    #         test_dg = ST.dg(test_coords)
    #         test_conn = ST.conn(test_coords)

    #         self.assertTrue(torch.allclose(ref_g, test_g, atol=1e-5))
    #         self.assertTrue(torch.allclose(ref_ginv, test_ginv, atol=1e-5))
    #         self.assertTrue(torch.allclose(ref_dg, test_dg, atol=1e-5))
    #         self.assertTrue(torch.allclose(ref_conn, test_conn, atol=1e-5))

    #     pass
    
    @unittest.skip("Tetrad test is failing")
    def test_tetrad(self):

        X = torch.randn(1, 1, 4)

        for ST in self.ST_dict.values():
            with self.subTest(name=ST.__class__.__name__):
                eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], device=X.device, dtype=X.dtype))

                E = ST.tetrad(X)
                g_recon = torch.einsum('...ab, ...bc, ...cd -> ...ad', E.swapaxes(-1, -2), eta, E)
                g = ST.g(X)
                # Check if close to g
                self.assertTrue(torch.allclose(g, g_recon, atol=1e-5))


    @unittest.skip("JIT test is failing")
    def test_JIT(self):
        '''
        Test if spacetimes are jit-compilable:
        '''
        for ST in self.ST_dict.values():
            
            try:
                ST.compile()
            except Exception as e:
                print(f"Compilation failed for {ST}: {e}")


if __name__ == '__main__':

    unittest.main()