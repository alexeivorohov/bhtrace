import torch

import sys
sys.path.append('.')
from bhtrace import SphericallySymmetric, Photon, HTracer

# Generating test points

ts = [0]
rs = [20]
ths = [torch.pi/2]
phs = [0, 3]
N_test_p = len(ts)*len(rs)*len(ths)*len(phs)

X0 = torch.zeros(N_test_p, 4)

i = 0
for t in ts:
    for r in rs:
        for th in ths:
            for ph in phs:
                X0[i, :] = torch.Tensor([t, r, th, ph])
                i += 1

P0 = torch.zeros(N_test_p, 4)
P0[:, 0] = 0.6
P0[:, 1] = -0.8


# Setting up tracer:

schw = lambda r: 1 - 2/r
schw_r = lambda r: 2*torch.pow(r, -2)

SchwST = SphericallySymmetric(f=schw, f_r=schw_r)

Phot = Photon(SchwST)

tracer = HTracer()
tracer.particle_set(Phot)

Xs, Ps = tracer.trace(X0, P0, 2e-5, 200, 5e-2)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(Xs[:, 0, 0], Xs[:, 0, 1], c='blue')  # Plot the trajectory
plt.plot(Xs[:, 1, 0], Xs[:, 1, 1], c='red')
# plt.title('Trajectory in Flat Space')
plt.xlabel('R')
plt.ylabel('T')
plt.axhline(0, color='black',linewidth=0.5, ls='--')  # Raxis
plt.axvline(0, color='black',linewidth=0.5, ls='--')  # Taxis
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
# plt.xlim(-1.5, 1.5)  # Set limits for x-axis
# plt.ylim(-1.5, 1.5)  # Set limits for y-axis
plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
plt.show()





