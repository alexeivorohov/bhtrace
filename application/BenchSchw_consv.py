import torch
import time
import pickle

import sys
import matplotlib.pyplot as plt
sys.path.append('.')

from bhtrace.geometry import SphericallySymmetric, KerrSchild, Photon
from bhtrace.functional import cart2sph, sph2cart

import functools
from matplotlib.animation import FuncAnimation


Ni = 760
with open('application/tr_ks_bench_{}.pkl'.format(Ni), 'rb') as file:
    res1 = pickle.load(file)
with open('application/tr2_bench_{}.pkl'.format(Ni), 'rb') as file:
    res2 = pickle.load(file)

Nt = res1['X'].shape[0]


spacetimes = [KerrSchild(a=0.0, m=1.0, Q=0.0), SphericallySymmetric()]
particles = [Photon(ST) for ST in spacetimes]
results = [res1, res2]

n_range = range(Ni)
t_range = [t+1 for t in range(Nt-1)]

E_1 = torch.Tensor([[particles[0].Hmlt(res1['X'][t, n, :], res1['P'][t, n, :]) for n in n_range] for t in t_range])
E_2 = torch.Tensor([[particles[1].Hmlt(res2['X'][t, n, :], res2['P'][t, n, :]) for n in n_range] for t in t_range])



# Create a figure and axes
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
plt_bins = torch.Tensor([0.0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
# plt_bins_= torch.Tensor([range(len(plt_bins-1))])
plt_alpha = 0.7
plt_edge_c = 'black'

F1 = torch.log(abs(E_1)+1.0)
F2 = torch.log(abs(E_2)+1.0)

ax[0].imshow(F1)
ax[1].imshow(F2)

# def update(frame):
#     ax[0].cla()  # Clear the previous histogram for E_1
#     ax[1].cla()  # Clear the previous histogram for E_2
    
#     dens1, bins1 = torch.histogram(torch.Tensor(E_1[frame]), bins=plt_bins, density=True)
#     dens2, bins2 = torch.histogram(torch.Tensor(E_2[frame]), bins=plt_bins, density=True)

#     # Plotting histograms for current frame
#     ax[0].bar(dens1, color='blue', alpha=plt_alpha, edgecolor=plt_edge_c)
#     ax[0].grid('on')
#     ax[0].set_ylims([-0.05, 1.05])
    
#     ax[1].bar(dens2, color='blue', alpha=plt_alpha, edgecolor=plt_edge_c)
#     ax[1].grid('on')
#     ax[1].set_ylims([-0.05, 1.05])

#     plt.title(f'Energy distributions at step {frame+1}')


# ani = FuncAnimation(fig, update, frames=Nt-1, repeat=True)

plt.tight_layout()
plt.show()
