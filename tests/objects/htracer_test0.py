import torch
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('.')

from bhtrace.geometry import MinkowskiCart, Photon
from bhtrace.functional import net, sph2cart, cart2sph
from bhtrace.imaging import HTracer

# Spacetime definition:

ST = MinkowskiCart()
Phot = Photon(ST)

# Tracer summoning:

tracer = HTracer()
tracer.particle_set(Phot)


# Initial data

Ni = 16
D0 = 16
db = 5

X0, P0 = torch.zeros(Ni, 4), torch.zeros(Ni, 4)

X0[:, 1] = torch.ones(Ni)*D0
X0[:, 2] = torch.linspace(-1, 1, Ni)*db

P0[:, 0] = torch.ones(Ni)
P0[:, 1] = -torch.ones(Ni)


# Tracing

X_res, P_res = tracer.trace(X0, P0, 1e-5, 64, 4e-1)

# Coordinate transform

X_sph, P_sph = cart2sph(X_res, P_res)
X_plt, P_plt = sph2cart(X_sph, P_sph)

# Imaging - cartesian

fig2, ax2 = plt.subplots(1,1,figsize=(8,6))

ax2.quiver(X_plt[:, :, 1], X_plt[:, :, 2], P_plt[:, :, 1], P_plt[:, :, 2])

ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.grid('on')
plt.show()

# Passed!