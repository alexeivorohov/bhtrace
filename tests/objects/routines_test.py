import sys
sys.path.append('.')
from bhtrace import net, cart2sph, Photon, SphericallySymmetric

import matplotlib.pyplot as plt

import torch


schw = lambda r: 1 - 2/r
schw_r = lambda r: 2*torch.pow(r, -2)

SchwST = SphericallySymmetric(f=schw, f_r=schw_r)

Phot = Photon(SchwST)

# line test 

r, th, ph, vr, vth, vph  = cart2sph(net(type='line', rng=10, D0=10))

t0 = torch.zeros_like(r)
vt = torch.zeros_like(r)

X0 = torch.cat([t0, r, th, ph], axis=1)
P0 = torch.cat([vt, vr, vth, vph], axis=1)

print(X0.shape)
print(P0.shape)

gX = SchwST.g(X0)
dPt = torch.einsum('bi,bij,bj->b', P0, gX, P0)
P0[:, 0] = torch.sqrt(abs(dPt/gX[:, 0, 0]))
print(torch.einsum('bi,bij,bj->b', P0, gX, P0))








