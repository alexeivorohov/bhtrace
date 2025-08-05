import torch
import time
import os

import sys
sys.path.append('/home/alexey/Work/bhtrace-dev/')

import torch
import numpy as np
from matplotlib import pyplot as plt

from bhtrace.geometry import Photon, MinkowskiCart, KerrSchild, SchwSchild
from bhtrace.tracing import PTracer as Tracer

from bhtrace.functional import print_status_bar, net

from mwe_setup import *

N_grid = 16
# N_grid = 100
b = 15
X0, Y0, Z0 = net('line', rng=(N_grid, 0), X0=16.0, YZsize=[b, 0], YZ0=[b/2, 0])

Ni = X0.shape[0]

X0 = torch.stack([torch.zeros(Ni), X0, Y0, Z0], dim=1)
V0 = torch.zeros(Ni, 4)
V0[:, 1] = -torch.ones(Ni)
V0[:, 0] = torch.ones(Ni)

fig, ax = plt.subplots(1,1, figsize=(6,6))

ax.scatter(Y0, Z0)
ax.set_title(Ni)
plt.show() 

tracer = Tracer(eps=0.001)

for key, spacetime in spacetimes.items():
    photon = Photon(spacetime=spacetime)
    P0 = photon.GetNullMomentum(X0, V0)
    # P0 = V0

    # print(P0)
    # print(spacetime.g(X0))
    # print(photon.dHmlt(X0, P0, 0.01))

    tracer.forward(photon, X0, P0, T=4.0, nsteps=4)
    X, P = tracer.forward(photon, X0, P0, T=60.0, nsteps=128)
    tracer.save(f'{key}_2d_{Ni}.pkl', 'data/')

    print(photon.Hmlt(X, P))



