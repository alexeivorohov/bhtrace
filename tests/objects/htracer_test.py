import torch

import sys
sys.path.append('.')
from bhtrace import SphericallySymmetric, Photon, HTracer, net, sph2cart, cart2sph


# Setting up tracer:

schw = lambda r: 1 - 2/r
schw_r = lambda r: 2*torch.pow(r, -2)

SchwST = SphericallySymmetric(f=schw, f_r=schw_r)

Phot = Photon(SchwST)

tracer = HTracer()
tracer.particle_set(Phot)

# Preparing to trace

N_init = 16

r, th, ph, vr, vth, vph  = cart2sph(net(type='line', db=[-10, 10, 0, -10], rng=N_init, D0=10))

t0 = torch.zeros_like(r)
vt = torch.zeros_like(r)

X0 = torch.cat([t0, r, th, ph], axis=1)
P0 = torch.cat([vt, vr, vth, vph], axis=1)

# print(X0.shape)
# print(P0.shape)

gX = SchwST.g(X0)
dPt = torch.einsum('bi,bij,bj->b', P0, gX, P0)
P0[:, 0] = torch.sqrt(abs(dPt/gX[:, 0, 0]))
# print(torch.einsum('bi,bij,bj->b', P0, gX, P0))

# Tracing
X_res, P_res = tracer.trace(X0, P0, 1e-5, 256, 5e-8)


# Imaging
import matplotlib.pyplot as plt

xps = X_res[:, :, 1], X_res[:, :, 2], X_res[:, :, 3], P_res[:, :, 1], P_res[:, :, 2], P_res[:, :, 3]

# Xs, Ys, Zs, Vx, Vy, Vz = sph2cart(xps)

fig, ax = plt.subplots(1,1,figsize=(8,6))

for i in range(N_init):
    ax.plot(X_res[:, :, 1], X_res[:, :, 2], '.-', c='blue')  

plt.show()

# Validating results:

n_t = 15
Xi, Pi = X_res[:, n_t, :], P_res[:, n_t, :]
gi = SchwST.g(Xi)
print(torch.einsum('ti,tij,tj->t', Pi, gi, Pi))

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





