
from ..imagung import HTracer
from ..geometry import Particle
from ..functional import sph2cart, cart2sph

import torch

def slice_bs(
    particle: Particle, bs: torch.Tensor, Nsteps, T,
    D0 = 10, dPhi = 0, dTh = 0, V0=1,
    trajstyle=None, showic=False):
    '''
    Example scenario
    '''
    assert len(bs.shape) == 1

    Ni = bs.shape[0]

    T0 = torch.zeros(Ni)
    X0 = torch.ones(Ni)*D0
    Y0 = bs
    Z0 = torch.zeros(Ni)

    Pt0


    tracer = HTracer()
    tracer.particle_set(Phot)

    if linestyle == None:
        trajsty = ''


    plt.show



