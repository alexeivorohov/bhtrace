import torch
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('.')

from bhtrace.geometry import Photon
from bhtrace.geometry import KerrSchild
from bhtrace.functional import points_generate
from bhtrace.imaging import HTracer

# Preset

ST = KerrSchild(a=0.5,m=1)

gma0 = Photon(ST)
tracer = HTracer(lmbda_tol=0.4)
tracer.particle_set(gma0)

# Initial data

Ni = 8
D0 = 16
db = 30

X0, P0 = torch.zeros(Ni, 4), torch.zeros(Ni, 4)

X0[:, 1] = torch.ones(Ni)*D0
X0[:, 2] = torch.linspace(0, -1, Ni)*db

P0[:, 0] = torch.ones(Ni)
P0[:, 1] = -torch.ones(Ni)


# Calculation

X_res, P_res = tracer.trace(X0, P0, 1e-5, 100, 0.2, crit_log=True)


# Imaging - cartesian
fig2, ax2 = plt.subplots(1,1,figsize=(8,6))

X_plt, P_plt = X_res, P_res

ax2.quiver(X_plt[:, :, 1], X_plt[:, :, 2], P_plt[:, :, 1], P_plt[:, :, 2])
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.axis('equal')
ax2.grid('on')
plt.show()