import torch
import matplotlib.pyplot as plt


import os
import sys
sys.path.append('.')

from bhtrace.geometry import Photon
from bhtrace.geometry import MinkowskiSph, SphericallySymmetric
from bhtrace.functional import net, sph2cart, cart2sph
from bhtrace.imaging import HTracer

# Preset

f = lambda r: 1-2/r
f_r = lambda r: 2*torch.pow(r, -2)

# ST = SphericallySymmetric(f, f_r)
ST = MinkowskiSph()

gma0 = Photon(ST)
tracer = HTracer()
tracer.particle_set(gma0)

# Initial data

Ni = 16
D0 = 16
db = 10

X0, P0 = torch.zeros(Ni, 4), torch.zeros(Ni, 4)

X0[:, 1] = torch.ones(Ni)*D0
X0[:, 2] = torch.linspace(-1, 1, Ni)*db

P0[:, 0] = torch.ones(Ni)
P0[:, 1] = -torch.ones(Ni)

X0sph, P0sph = cart2sph(X0, P0)
P0sph = gma0.normp(X0sph, P0sph)
# Calculation

Res0 = tracer.solve(X0sph, P0sph, 30, 100)
X_res, P_res = Res0.ys[:, :, 0:4], Res0.ys[:, :, 4:]

# Imaging - cartesian
fig2, ax2 = plt.subplots(1,1,figsize=(8,6))

X_plt, P_plt = sph2cart(X_res, P_res)

ax2.quiver(X_plt[:, :, 1], X_plt[:, :, 2], P_plt[:, :, 1], P_plt[:, :, 2])
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.grid('on')
plt.show()