import torch
import time

import sys
sys.path.append('.')
from bhtrace.geometry import EffGeomSPH, Photon
from bhtrace.functional import cart2sph, sph2cart, net
from bhtrace.tracing import PTracer, NTracer
from bhtrace.electrodynamics import Maxwell, EulerHeisenberg, ModMax, Bardeen

import matplotlib.pyplot as plt


##################################
#  Electrodynamics preset:       #
##################################

# Constants and definitions
q = 0.5
q2 = q**2
h = 10
# a = 16*mu0*h
a = 16*4*torch.pi*h
a_10 = a/10

# Models
ED_dict = {
    'Maxwell': Maxwell(),
    'EulerHeisenberg_m': EulerHeisenberg(h=h),
    'EulerHeisenberg_e': EulerHeisenberg(h=h),
    'EulerHeisenberg_me': EulerHeisenberg(h=h)
    }

# Metric functions
f_dict = {
    'Maxwell': lambda r: 1.0 - 2.0*torch.pow(r, -1) + q2*torch.pow(r, -2),
    'EulerHeisenberg_m': lambda r: 1.0 - 2.0*torch.pow(r, -1) + q2*torch.pow(r, -2)*(1 - a_10*q2*q2*torch.pow(r, -4)), 
    'EulerHeisenberg_e': lambda r: 1.0 - 2.0*torch.pow(r, -1) + q2*torch.pow(r, -2),
    'EulerHeisenberg_me': lambda r: 1.0 - 2.0*torch.pow(r, -1) + q2*torch.pow(r, -2)*(1 - a_10*q2*q2*torch.pow(r, -4))
    }

df_dict = {
    'Maxwell': lambda r: 2.0*torch.pow(r, -2) - 2*q2*torch.pow(r, -3),
    'EulerHeisenberg_m': lambda r: 2.0*torch.pow(r, -2) - 2*q2*torch.pow(r, -3) + 6*a_10*torch.pow(r, -7)*q2**3,
    'EulerHeisenberg_e': lambda r: 2.0*torch.pow(r, -2) - 2*q2*torch.pow(r, -3),
    'EulerHeisenberg_me': lambda r: 2.0*torch.pow(r, -2) - 2*q2*torch.pow(r, -3) + 6*a_10*torch.pow(r, -7)*q2**3,
    }

# Field of a point charge
Er_dict = {
    'Maxwell': lambda r: q*torch.pow(r, -2),
    'EH': lambda r: q*torch.pow(r, -2) - a*q2*torch.pow(r, -6)
}

# fields
B_dict = {
    'Maxwell': lambda X: torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    'EulerHeisenberg_m': lambda X: torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    'EulerHeisenberg_e': lambda X: torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    'EulerHeisenberg_me': lambda X: torch.Tensor([0.0, 0.0, 0.0, 0.0])
    }

E_dict = {
    'Maxwell': lambda X: torch.Tensor([0.0, 0.0, 0.0, 0.0]),
    'EulerHeisenberg_m': lambda X: torch.Tensor([0.0, 0, 0.0, 0.0]),
    'EulerHeisenberg_e': lambda X: torch.Tensor([0.0, Er_dict['EH'](X[1]), 0.0, 0.0]),
    'EulerHeisenberg_me': lambda X: torch.Tensor([0.0, Er_dict['EH'](X[1]), 0.0, 0.0])
    }


 
########################################
# Initializing spacetimes and photons: #    
########################################

ST_dict = {}

for k in ED_dict.keys():

    ST_dict[k] = EffGeomSPH(
        ED=ED_dict[k],
        E=E_dict[k],
        B=B_dict[k],
        f=f_dict[k],
        f_r=df_dict[k]
        )


# Attaching particles:

P_dict = {}

for k in ED_dict.keys():

    P_dict[k] = Photon(ST_dict[k])

#######################################
# Initial conditions                  #
#######################################

rng = 100
b = 10
X0, Y0, Z0 = net('line', rng=(rng, 0), X0=20.0, YZsize=[b, 0], YZ0=[b/2, 0])

Ni = X0.shape[0]

X0 = torch.stack([torch.zeros(Ni), X0, Y0, Z0], dim=1)
P0 = torch.zeros(Ni, 4)
P0[:, 0] = torch.ones(Ni)
P0[:, 1] = -torch.ones(Ni)

X0sph, P0sph = cart2sph(X0, P0)


#######################################
# Perform tracing and save:           #
#######################################

tracer = PTracer()

SESSION_NAME = 'lensingHEv1'
lst = ['Maxwell', 'EulerHeisenberg_m', 'EulerHeisenberg_e', 'EulerHeisenberg_me']
for k in lst:

    P0sph_cov = torch.zeros(Ni, 4)
    for i in range(Ni):
        P0sph_cov[i, :] = P_dict[k].GetNullMomentum(X0sph[i, :], P0sph[i, 1:])

    tracer.forward(P_dict[k], X0sph, P0sph_cov, T=60.0, nsteps=128)
    tracer.save(
        '{}_{}_{}.pkl'.format(SESSION_NAME, Ni, k),
        directory='application/')

