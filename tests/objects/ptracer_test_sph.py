import torch
import time

import sys
sys.path.append('.')
from bhtrace.geometry import SphericallySymmetric, MinkowskiSph, Photon
from bhtrace.functional import points_generate, cart2sph, sph2cart
from bhtrace.imaging import PTracer

import matplotlib.pyplot as plt

# Preset

ST = SphericallySymmetric()
# ST = MinkowskiSph()

gma0 = Photon(ST)
tracer = PTracer()
tracer.particle_set(gma0)

# Initial data

Ni = 16
D0 = 16
db = 20

X0, P0 = torch.zeros(Ni, 4), torch.zeros(Ni, 4)

X0[:, 1] = torch.ones(Ni)*D0
X0[:, 2] = torch.linspace(0.5, 1, Ni)*db

P0[:, 0] = torch.ones(Ni)
P0[:, 1] = -torch.ones(Ni)

X0, P0 = cart2sph(X0, P0)

for i in range(Ni):
    P0[i, :] = gma0.GetNullMomentum(X0[i, :], P0[i, 1:])

# Calculation

X_res, P_res = tracer.trace(X0, P0, nsteps=128, dt=0.3)


# Imaging - cartesian
fig2, ax2 = plt.subplots(1,1,figsize=(8,6))

# X_plt, P_plt = X_res, P_res
X_plt, P_plt = sph2cart(X_res, P_res)

# ax2.quiver(X_plt[:, :, 1], X_plt[:, :, 2], P_plt[:, :, 1], P_plt[:, :, 2])
ax2.plot(X_plt[:, :, 1], X_plt[:, :, 2])
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.axis('equal')
ax2.grid('on')
plt.show()