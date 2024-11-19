import torch
import unittest

import sys
sys.path.append('.')

from bhtrace.geometry import Particle, Photon,\
    MinkowskiCart, MinkowskiSph, SphericallySymmetric

from bhtrace.functional import sph2cart, cart2sph, points_generate


class TestPhoton(unittest.TestCase):

    def setUp(self):

        self.probe_ST = MinkowskiSph()
        self.photon = Photon(self.probe_ST)
        self.atol = 1e-6
        self.rtol = 1e-6
        self.eps = 1e-4

    def test_Momentum(self):

        X = torch.tensor([2, 3, 3, 3], dtype=torch.float32)
        v = torch.tensor([0.8, 0.6, 0.0])

        P = self.photon.GetNullMomentum(X, v)
        ginv = self.probe_ST.ginv(X)
        normP = (ginv @ P) @ P

        exp_normP = torch.Tensor([0.0])

        v_ = self.photon.GetDirection(X, P)

        self.assertTrue(torch.isclose(normP, exp_normP, atol=self.atol, rtol=self.rtol),
        'GetNullMomentum returned momentum of incorrect norm: {}'.format(normP))

        self.assertTrue(torch.allclose(v, v_, atol=self.atol, rtol=self.rtol),
        'Direction reconstructed unsucessfully from generated momentum')


    def test_Hmlt(self):

        X = torch.tensor([2, 3, 3, 3], dtype=torch.float32)
        v = X[1:]
        P = self.photon.GetNullMomentum(X, v)

        H = self.photon.Hmlt(X, P)

        expH = torch.Tensor([0.0])
        

        self.assertTrue(H.shape == expH.shape, 
        'Incorrect H(X, P) shape returned: {}'.format(H.shape))

        self.assertTrue(torch.isclose(H, expH, atol=self.atol, rtol=self.rtol),
        'Incorrect H(X, P) value: {}'.format(H))


    def test_dHmlt(self):

        X = torch.tensor([2, 3, 3, 3],  dtype=torch.float32)
        v = X[1:]
        P = self.photon.GetNullMomentum(X, v)

        dH = self.photon.dHmlt(X, P, self.eps)
        exp_dH = torch.Tensor([0.0, 0.0, 0.0, 0.0])

        self.assertTrue(H.shape == expH.shape, 
        'Incorrect shape returned: {}'.format(H.shape))

        # test H(x1)=X(x2, p1)+dH(x2, p1)*(x1-x2) 


if __name__ == '__main__':
    unittest.main()


