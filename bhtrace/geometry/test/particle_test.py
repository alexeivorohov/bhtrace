import torch
import unittest

import sys
sys.path.append('.')

from bhtrace.geometry import Particle, Photon,\
    MinkowskiCart, MinkowskiSph, SphericallySymmetric

from bhtrace.geometry.spacetimes_cart import KerrSchild


# from bhtrace.functional import sph2cart, cart2sph, points_generate


# [] Should be generalized for other particles and spacetimes
# [] More evaluation points, may be also check spacetime symmetries

class TestPhoton(unittest.TestCase):

    def setUp(self):
        self.spacetimes = {
            'minkowski_cart': MinkowskiCart(),
            'minkowski_sph': MinkowskiSph(),
            'spherically_symmetric': SphericallySymmetric(),
            'kerr_schild_01': KerrSchild(a=0.1),
            'kerr_schild_05': KerrSchild(a=0.5),
            'kerr_schild_09': KerrSchild(a=0.9),
        }
        self.photons = {name: Photon(st) for name, st in self.spacetimes.items()}
        self.atol = 1e-5
        self.rtol = 1e-5
        self.eps = 1e-4

    def test_Momentum(self):
        '''
        Test momentum calculation:
        - norm correctness
        - reconstruction of velocity correctness

        '''
        for name, photon in self.photons.items():
            with self.subTest(spacetime=name):
                # Points and velocities
                X = torch.tensor([2, 3, 3, 3], dtype=torch.float32)
                v = torch.tensor([0.8, 0.6, 0.0])

                # Calculate impulse
                P = photon.GetNullMomentum(X, v)

                # Calculate impulse norm
                ginv = photon.spacetime.ginv(X)
                normP = (ginv @ P) @ P

                # Expected norm
                exp_normP = torch.tensor(0.0, dtype=X.dtype)

                # Reconstruct velocity
                v_ = photon.GetDirection(X, P)

                # Test
                self.assertTrue(torch.isclose(normP, exp_normP, atol=self.atol, rtol=self.rtol),
                f'GetNullMomentum returned momentum of incorrect norm: {normP} for {name}')

                self.assertTrue(torch.allclose(v, v_, atol=self.atol, rtol=self.rtol),
                f'Direction reconstructed unsucessfully from generated momentum for {name}')


    def test_Hmlt(self):
        '''
        Test hamiltonian of particle:
        - Shape correctnes
        - Value correctnes
        '''
        for name, photon in self.photons.items():
            with self.subTest(spacetime=name):
                # Set up coordinates, velocities and impulses
                X = torch.tensor([2, 3, 3, 3], dtype=torch.float32)
                v_unnorm = torch.tensor([1., 1., 1.], dtype=X.dtype)
                v = v_unnorm / torch.norm(v_unnorm)
                P = photon.GetNullMomentum(X, v)

                # Calculate hamiltonian
                H = photon.Hmlt(X, P)

                # Expected hamiltonian
                expH = torch.tensor(0.0, dtype=X.dtype)
                
                self.assertTrue(H.shape == expH.shape, 
                f'Incorrect H(X, P) shape returned: {H.shape} for {name}')

                self.assertTrue(torch.isclose(H, expH, atol=self.atol, rtol=self.rtol),
                f'Incorrect H(X, P) value: {H} for {name}')

    def test_dHmlt(self):
        '''
        Test hamiltonian derivative:
        - Shape correctness
        - Check taylor expansion
        '''
        for name, photon in self.photons.items():
            with self.subTest(spacetime=name):
                X = torch.tensor([2, 3, 3, 3],  dtype=torch.float32)
                v_unnorm = torch.tensor([1., 1., 1.], dtype=X.dtype)
                v = v_unnorm / torch.norm(v_unnorm)
                P = photon.GetNullMomentum(X, v)

                dH = photon.dHmlt(X, P, self.eps)

                # test H(X + dX) approx dH/dX * dX
                dX = torch.randn_like(X) * 1e-5
                H1 = photon.Hmlt(X + dX, P)
                H_approx = torch.dot(dH, dX)
                
                self.assertTrue(torch.isclose(H1, H_approx, atol=self.atol, rtol=self.rtol),
                                f'Incorrect dHmlt for {name}. H(X+dX)={H1}, dH*dX={H_approx}')

    def test_save_load(self):
        '''
        Test if particle can be saved and loaded
        '''
        for name, photon in self.photons.items():
            with self.subTest(spacetime=name):
                state = photon.state()
                loaded_photon = Particle.from_dict(state.copy())
                new_state = loaded_photon.state()

                # TODO: Fix spacetime serialization
                state.pop('spacetime')
                new_state.pop('spacetime')
                
                self.assertEqual(state, new_state,
                                 f'State mismatch for {name}: \n original {state}:,\n loaded: {new_state}'
                                 ) 


if __name__ == '__main__':
    unittest.main()


