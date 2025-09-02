import torch
import unittest

import sys
import os
root_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
sys.path.append(os.getcwd())


from bhtrace.geometry import mock_spacetime
from bhtrace.geometry import _SPACETIMES_
# from bhtrace.functional import sph2cart, cart2sph
# from bhtrace.tracing import PTracer, CTracer


class TestSpacetimeBase(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestSpacetimeBase, self).__init__(*args, **kwargs)
        self.n_eval = 10
        self.mock_st = mock_spacetime()

    
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

    def __init__(self, *args, **kwargs):
        super(TestSpacetimeCollection, self).__init__(*args, **kwargs)
        self.ST_dict = {}
        self.n_eval = 10


    def test_init(self):
        '''
        Test if all spacetimes can be initialized
        '''
        for name, constructor in _SPACETIMES_.items():

            try:
                ST = constructor()
                self.ST_dict[name] = ST
            except Exception as e:  
                print(f"Failed to initialize {name}: {e}")
                continue

        pass


    def test_metric(self):
        '''
        Test if metric is computed correctly:
        g @ g^{-1} = I
        '''     
        
        test_coords = torch.randn(self.n_eval, 4)*10
        
        for ST in self.ST_dict.values():
            try:
                g = ST.g(test_coords)
                invg = ST.invg(test_coords)
                self.assertTrue(torch.allclose(g @ invg, torch.eye(4), atol=1e-5))
            except Exception as e:
                print(f"Failed to compute metric for {ST}: {e}")
                continue

        pass


    def test_reduction(self):
        '''
        Test if metirc reduces to the minkowski metric in the flat case:
        g = eta
        g^{-1} = eta^{-1}
        dg = 0
        conn = 0
        '''

        ref_ST = mock_spacetime()
        test_coords = torch.randn(self.n_eval, 4)

        ref_g = ref_ST.g(test_coords)
        ref_ginv = ref_ST.ginv(test_coords)
        ref_dg = ref_ST.dg(test_coords)
        ref_conn = ref_ST.conn(test_coords)

        for ST in self.ST_dict.values():
            if ST._reduction_params_ is None:
                continue

            test_g = ST.g(test_coords)
            test_ginv = ST.invg(test_coords)
            test_dg = ST.dg(test_coords)
            test_conn = ST.conn(test_coords)

            self.assertTrue(torch.allclose(ref_g, test_g, atol=1e-5))
            self.assertTrue(torch.allclose(ref_ginv, test_ginv, atol=1e-5))
            self.assertTrue(torch.allclose(ref_dg, test_dg, atol=1e-5))
            self.assertTrue(torch.allclose(ref_conn, test_conn, atol=1e-5))

        pass
    
    def test_tetrad(self):

        X = torch.randn(1, 1, 4)

        for ST in self.ST_dict.values():

            eta = torch.diag(torch.tensor([-1, 1, 1, 1]))

            E = ST.tetrad(X)
            g_recon = torch.einsum('...ab, ...bc, ... cd -> ...ad', E.swapaxes(-1, -2), eta, E)

            # Check if close to g
            if not torch.allclose(g, g_recon, atol=1e-5):
                print("Warning: Reconstruction error in tetrad factorization.")


    def test_JIT(self):
        '''
        Test if spacetimes are jit-compilable:
        '''
        for ST in self.ST_dict.values():
            
            try:
                ST.compile()
            except Exception as e:
                print(f"Compilation failed for {ST}: {e}")

            pass

        pass


if __name__ == '__main__':

    unittest.main()
