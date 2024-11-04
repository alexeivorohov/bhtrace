import torch

import sys
sys.path.append('.')

from bhtrace.geometry import Particle, Photon,\
    MinkowskiCart, MinkowskiSph, SphericallySymmetric

from bhtrace.functional import sph2cart, cart2sph, points_generate


def particle_test(Particle: Particle, ST, X, P, prnt=False):

    atol = 1e-6
    rtol = 1e-6

    pcle = Particle(ST)

    # Test impulse norm:

    gX = pcle.Spacetime.g(X)

    mu0 = torch.einsum('bi, bij, bj->b', P, gX, P)
    P_ = pcle.normp(X, P)
    mu_ = torch.einsum('bi, bij, bj->b', P_, gX, P_)

    tst1 = torch.allclose(mu_, mu0, atol=atol, rtol=rtol)
    print('Norm test result: {}'.format(tst1))

    if prnt:
        print(mu0)
        print(mu_)

    # Test Hamiltonian

    # H = Phot.Hmlt(X, P)
    # print(H.shape)

    # pre = torch.ones_like(X)
    # DVec = torch.einsum('bi,ij->bij', pre, torch.eye(4))

    # dH = Phot.dHmlt(X, P, DVec, 2e-5)

    # print(dH)

    return (tst1)

# Coords
ts = [0]

rs = [2, 20]
ths = [0, 1]
phs = [0, 1.5, 3.14]

xs = [2, -2]
ys = [2, -2]
zs = [2, -3]

Xcart = points_generate(ts, xs, ys, zs)
Xsph = points_generate(ts, rs, ths, phs)

Pcart = torch.zeros_like(Xcart)
Pcart[:, 0] = torch.ones(Pcart.shape[0])
Pcart[:, 1] = torch.ones(Pcart.shape[0])

Psph = torch.zeros_like(Xsph)
Psph[:, 0] = torch.ones(Psph.shape[0])
Psph[:, 1] = torch.ones(Psph.shape[0])

# Spacetimes

MCart = MinkowskiCart()
MSph = MinkowskiSph()

particle_test(Photon, MCart, Xcart, Pcart)
particle_test(Photon, MSph, Xsph, Psph)


