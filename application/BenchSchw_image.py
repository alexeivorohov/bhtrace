import torch
import time
import pickle

import sys
import matplotlib.pyplot as plt
sys.path.append('.')

from bhtrace.geometry import SphericallySymmetric, Photon
from bhtrace.functional import cart2sph, sph2cart
from bhtrace.radiation import thin_disk, i2_r


Ni = 760
with open('application/tr_ks_bench_{}.pkl'.format(Ni), 'rb') as file:
    res1 = pickle.load(file)
with open('application/tr2_bench_{}.pkl'.format(Ni), 'rb') as file:
    res2 = pickle.load(file)

ST = SphericallySymmetric()
phot = Photon(ST)
f = ST.f
f_r = ST.f_r

# X1cart, P1cart = sph2cart(res1['X'], res1['P'])
X1cart, P1cart = res1['X'], res1['P']
X2cart, P2cart = sph2cart(res2['X'], res2['P'])
Y0, Z0 = X1cart[0, :, 2], X1cart[0, :, 3]


mask = torch.isclose(Z0, torch.Tensor([1.0]), atol=1e0, rtol=1e0)

fig, axs = plt.subplot_mosaic([['Tr1', 'Tr2']], figsize=(16, 8))
ax = axs['Tr1']
ax.plot(X1cart[:, mask, 1], X1cart[:, mask, 2], '.-')
ax.axis('equal')
circle1 = plt.Circle((0, 0), 2, lw=2, ls='--', color='m', fill=False)
ax.add_patch(circle1)
ax.set_title('Tr1')
ax.set_xlabel('$Y/M$')
ax.set_ylabel('$Z/M$')
ax.set_xlim([-10, 20])
ax.set_ylim([-15, 15])

ax = axs['Tr2']
ax.plot(X2cart[:, mask, 1], X2cart[:, mask, 2], '.-')
circle2 = plt.Circle((0, 0), 2, lw=2, ls='--', color='m', fill=False)
ax.add_patch(circle2)
ax.set_title('Tr2')
ax.set_xlabel('$Y/M$')
ax.set_ylabel('$Z/M$')
ax.axis('equal')
ax.set_xlim([-10, 20])
ax.set_ylim([-15, 15])

plt.show()
norm_v = torch.Tensor([1, 0, 5])


I2 = lambda r: i2_r(r, f)

F1 = thin_disk(
    R = torch.linalg.vector_norm(X1cart[:, :, 1:], dim=-1, ord=1), 
    X = X1cart[:, :, 1], 
    Y = X1cart[:, :, 2], 
    Z = X1cart[:, :, 3], 
    norm_v = norm_v, 
    I_r = I2, 
    r_H = 2.0)

F2 = thin_disk(
    R = res2['X'][:, :, 1], 
    X = X2cart[:, :, 1], 
    Y = X2cart[:, :, 2], 
    Z = X2cart[:, :, 3], 
    norm_v = norm_v, 
    I_r = I2, 
    r_H = 2.0)

F1 = torch.log(F1+0.001)
F2 = torch.log(F2+0.001)

fig, axs = plt.subplot_mosaic([['Tr1', 'Tr2']], figsize=(16, 8))
ax = axs['Tr1']
cm = ax.scatter(Y0, Z0, marker='h', c=F1.view(-1, 1), cmap='hot')
ax.axis('equal')
circle1 = plt.Circle((0, 0), 2, lw=2, ls='--', color='m', fill=False)
ax.add_patch(circle1)
ax.set_title('Tr1')
ax.set_xlabel('$Y/M$')
ax.set_ylabel('$Z/M$')
plt.colorbar(cm)

ax = axs['Tr2']
cm = ax.scatter(Y0, Z0, marker='h', c=F2.view(-1, 1), cmap='hot')
circle2 = plt.Circle((0, 0), 2, lw=2, ls='--', color='m', fill=False)
ax.add_patch(circle2)
plt.colorbar(cm)
ax.set_title('Tr2')
ax.set_xlabel('$Y/M$')
ax.set_ylabel('$Z/M$')
ax.axis('equal')

plt.show()