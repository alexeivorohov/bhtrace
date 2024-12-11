import torch
import time

import sys
sys.path.append('.')
from bhtrace.geometry import SphericallySymmetric, EffGeomSPH, Photon
from bhtrace.functional import cart2sph, sph2cart, net
from bhtrace.electrodynamics import Maxwell, EulerHeisenberg, ModMax, Bardeen

import matplotlib.pyplot as plt
from bhtrace.radiation import thin_disk, i2_r
import pickle

#########################################
# Data loading                          #
#########################################
Ni = 40
ED = ['Maxwell', 'EulerHeisenberg_m', 'EulerHeisenberg_e', 'EulerHeisenberg_me']
SESSION_NAME = 'lensingHEv1'

res = {}

for ED_ in ED:
    with open('application/{}_{}_{}.pkl'.format(SESSION_NAME, Ni, ED_), 'rb') as file:
        res[ED_] = pickle.load(file)


#########################################
# Electrodynamics presets               #
#########################################

pass


#########################################
# Coordinate transformations            #
#########################################

XP_dict = {}
R_dict = {}

for k in ED:
    
    X, P = res[k]['X'], res[k]['P']
    R_dict[k] = X[:, :, 1]
    XP_dict[k] = sph2cart(X, P)


#########################################
# Plot trajectories                     #
#########################################

# ED_plt = [['Maxwell', 'EulerHeisenberg_m'], ['EulerHeisenberg_e', 'EulerHeisenberg_me']]
# fig_h = 6
# fig, axs = plt.subplot_mosaic(ED_plt, figsize=(fig_h*2, fig_h*2))

# for k in ED:
    
#     X = XP_dict[k][0][:, :, 1]
#     Y = XP_dict[k][0][:, :, 2]
    
#     ax = axs[k]
    
#     ax.plot(X, Y, 'k-')

#     ax.axis('equal')
#     ax.set_xlim([-10, 20])
#     ax.set_ylim([-15, 15])

#     ax.grid('on')
#     ax.set_title(k)
#     ax.set_xlabel('$Y/M$')
#     ax.set_ylabel('$Z/M$')

# plt.show()

#########################################
# Plot lensing                          #
#########################################

def last_non_nan(X):
    
    n = X.shape[0] - 1
    n = X.shape[0] - 1
    mask = torch.isnan(X)
    nonnan = 0
    for k in range(n):
        if mask[n-k] == False:
            nonnan = X[n-k]
            break
    return nonnan


fig, ax = plt.subplots(1,1, figsize=(6, 6))

for k in ED:

    phi_all = res[k]['X'][:, :, 3]
    phi = torch.zeros(Ni)
    
    for n in range(Ni):
        phi[n] = last_non_nan(phi_all[:, n])

    b = XP_dict[k][0][0, :, 2]
    
    ax.plot(b, phi/2/torch.pi)
    ax.grid('on')
    ax.set_ylabel('$n$, number of turns')
    ax.set_xlabel('$b$, impact factor')

plt.legend(ED)
plt.show()