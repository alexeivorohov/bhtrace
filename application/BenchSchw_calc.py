import torch
import time

import sys
sys.path.append('.')
from bhtrace.geometry import SphericallySymmetric, MinkowskiSph, Photon
from bhtrace.functional import cart2sph, sph2cart, net
from bhtrace.tracing import PTracer, CTracer


import matplotlib.pyplot as plt

# Spacetime, for which connection symbols are known
ST = SphericallySymmetric()
# ST = MinkowskiSph()
phot = Photon(ST)

# Tracers:
tr1 = CTracer()
tr2 = PTracer()

# Initial conditions:


X0, Y0, Z0 = net('circle', rng=(40, 0), X0=20.0, YZsize=[20, 20])

print(X0.shape)
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.plot(Y0, Z0, '.')
ax.grid('on')
ax.axis('equal')
plt.show()


Ni = X0.shape[0]

X0 = torch.stack([torch.zeros(Ni), X0, Y0, Z0], dim=1)
P0 = torch.zeros(Ni, 4)
P0[:, 0] = torch.ones(Ni)
P0[:, 1] = -torch.ones(Ni)

X0sph, P0sph = cart2sph(X0, P0)

P0sph_cov = torch.zeros_like(P0sph)

for i in range(Ni):
    P0sph_cov[i, :] = phot.GetNullMomentum(X0sph[i, :], P0sph[i, 1:])


X1_res, P1_res = tr1.forward(phot, X0sph, P0sph, T=30.0, nsteps=64)
tr1.save('tr1_bench_{}.pkl'.format(Ni), directory='application/')

X2_res, P2_res = tr2.forward(phot, X0sph, P0sph_cov, T=30.0, nsteps=64)
tr2.save('tr2_bench_{}.pkl'.format(Ni), directory='application/')