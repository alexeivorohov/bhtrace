import torch
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('.')

from bhtrace.geometry import MinkowskiSph, MinkowskiCart, Photon
from bhtrace.functional import net, sph2cart, cart2sph
from bhtrace.imaging import NTracer



# Setting up tracer:

schw = lambda r: 1 
schw_r = lambda r: 0

ST = MinkowskiCart()

Phot = Photon(ST)

tracer = NTracer()
tracer.particle_set(Phot)

# Initial Data

Ni = 4
D0 = 16
db = 5

X0, P0 = torch.zeros(Ni, 4), torch.zeros(Ni, 4)

X0[:, 1] = torch.ones(Ni)*D0
X0[:, 2] = torch.linspace(-1, 1, Ni)*db

P0[:, 0] = torch.ones(Ni)
P0[:, 1] = - torch.ones(Ni)

# fig, ax = plt.subplots(1,1,figsize=(8,6))

# ax.plot(X0[:, 1], X0[:, 2], '.', c='r')
# ax.grid('on')
# plt.show()

X0sph, P0sph = cart2sph(X0, P0)


# Tracing
X_res, P_res = tracer.trace(X0, P0, 1e-5, 64, 5e-2)

# Imaging - cartesian
fig2, ax2 = plt.subplots(1,1,figsize=(8,6))

X_plt, P_plt = X_res, P_res
# X_plt, P_plt = sph2cart(X_res, P_res)

ax2.quiver(X_plt[:, :, 1], X_plt[:, :, 2], P_plt[:, :, 1], P_plt[:, :, 2])
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.grid('on')
plt.show()