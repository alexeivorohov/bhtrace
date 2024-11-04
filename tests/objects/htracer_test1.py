import torch
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('.')

from bhtrace.geometry import MinkowskiSph, SphericallySymmetric, Photon
from bhtrace.functional import net, sph2cart, cart2sph
from bhtrace.imaging import HTracer



# Setting up tracer:

schw = lambda r: 1-2/r
schw_r = lambda r: 2*torch.pow(r, -2)

ST = SphericallySymmetric(schw, schw_r)

Phot = Photon(ST)

tracer = HTracer()
tracer.particle_set(Phot)


# Initial Data

Ni = 4
D0 = 6
db = 5

X0, P0 = torch.zeros(Ni, 4), torch.zeros(Ni, 4)

X0[:, 1] = torch.ones(Ni)*D0
X0[:, 2] = torch.linspace(-1, 1, Ni)*db

P0[:, 0] = torch.ones(Ni)
P0[:, 1] = -torch.ones(Ni)

# fig, ax = plt.subplots(1,1,figsize=(8,6))

# ax.plot(X0[:, 1], X0[:, 2], '.', c='r')
# ax.grid('on')
# plt.show()

X0sph, P0sph = cart2sph(X0, P0)
P0sph = Phot.normp(X0sph, P0sph)



# Tracing
X_res, P_res = tracer.trace(X0sph, P0sph, 1e-5, 64, 1e-1)


# Imaging - spherical
# fig, ax = plt.subplots(1,1,figsize=(8,6))

# for i in range(Ni):
#     ax.plot(X_res[:, i, 3], X_res[:, i, 1], '.-', c='blue')  

# ax.set_xlabel('phi')
# ax.set_ylabel('R')
# ax.grid('on')
# plt.show()


# Imaging - cartesian
fig2, ax2 = plt.subplots(1,1,figsize=(8,6))

X_plt, P_plt = sph2cart(X_res, P_res)

ax2.quiver(X_plt[:, :, 1], X_plt[:, :, 2], P_plt[:, :, 1], P_plt[:, :, 2])
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.grid('on')
plt.show()


# Validating results:

# n_t = 1
# Xi, Pi = X_res[:, n_t, :], P_res[:, n_t, :]
# gi = ST.g(Xi)
# norm = torch.einsum('ti,tij,tj->t', Pi, gi, Pi)
# print(torch.allclose(norm, torch.zeros_like(norm), 1e-2))

# plt.plot(Xs[:, 1, 0], Xs[:, 1, 1], c='red')
# plt.title('Trajectory in Flat Space')
# plt.xlabel('R')
# plt.ylabel('T')
# plt.axhline(0, color='black',linewidth=0.5, ls='--')  # Raxis
# plt.axvline(0, color='black',linewidth=0.5, ls='--')  # Taxis
# plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
# plt.legend()
# # plt.xlim(-1.5, 1.5)  # Set limits for x-axis
# # plt.ylim(-1.5, 1.5)  # Set limits for y-axis
# plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
# plt.show()
