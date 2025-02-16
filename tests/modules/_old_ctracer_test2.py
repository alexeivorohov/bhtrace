import torch
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('.')

from bhtrace.geometry import MinkowskiSph, SphericallySymmetric, Photon
from bhtrace.functional import net, sph2cart, cart2sph
from bhtrace.tracing import CTracer

# Preset

f = lambda r: 1-2/r
f_r = lambda r: 2*torch.pow(r, -2)

# ST = SphericallySymmetric(f, f_r)
ST = MinkowskiSph()
gma0 = Photon(ST)

tracer = CTracer()


# Initial data

Ni = 8
D0 = 16
db = 10

X0, P0 = torch.zeros(Ni, 4), torch.zeros(Ni, 4)

X0[:, 1] = torch.ones(Ni)*D0
X0[:, 2] = torch.linspace(-1, 1, Ni)*db

P0[:, 0] = torch.ones(Ni)s
P0[:, 1] = -torch.ones(Ni)

X0sph, P0sph = cart2sph(X0, P0)

# Calculation

X_res, P_res = tracer.forward(gma0, X0sph, P0sph, T=30, nsteps=100)

# Imaging - cartesian
fig2, ax2 = plt.subplots(1,1,figsize=(8,6))

X_plt, P_plt = sph2cart(X_res, P_res)

ax2.quiver(X_plt[:, :, 1], X_plt[:, :, 2], P_plt[:, :, 1], P_plt[:, :, 2])
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.grid('on')
plt.show()