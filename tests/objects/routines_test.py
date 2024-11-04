import sys
sys.path.append('.')

from bhtrace.functional import net, cart2sph, sph2cart
from bhtrace.geometry import Photon, SphericallySymmetric

import matplotlib.pyplot as plt
import torch


schw = lambda r: 1 - 2/r
schw_r = lambda r: 2*torch.pow(r, -2)

SchwST = SphericallySymmetric(f=schw, f_r=schw_r)

Phot = Photon(SchwST)

# Sphere test

Ni = 4

X0 = torch.zeros(Ni, 4) 
X0[:, 1] = torch.ones(Ni)
X0[:, 2] = torch.rand(Ni)*torch.pi
X0[:, 3] = -torch.pi+torch.rand(Ni)*torch.pi*2

P0 = torch.zeros(Ni, 4)
P0[:, 1] = torch.ones(Ni)

Xcart, Pcart = sph2cart(X0, P0)
X1, P1 = cart2sph(Xcart, Pcart)

print(torch.allclose(X1, X0, atol=1e-6))

# Line test

Ni=21

X0 = torch.zeros(Ni, 4) 
X0[:, 1] = torch.linspace(-1, 1, Ni)*5
X0[:, 2] = torch.ones(Ni)*5
X0[:, 3] = torch.zeros(Ni)

P0 = torch.zeros(Ni, 4)
P0[:, 1] = torch.ones(Ni)

Xsph, Psph = cart2sph(X0, P0)
X1, P1 = sph2cart(Xsph, Psph)

print(torch.allclose(X0, X1, atol=1e-6))






# gX = SchwST.g(X0)
# dPt = torch.einsum('bi,bij,bj->b', P0, gX, P0)
# P0[:, 0] = torch.sqrt(abs(dPt/gX[:, 0, 0]))
# print(torch.einsum('bi,bij,bj->b', P0, gX, P0))








