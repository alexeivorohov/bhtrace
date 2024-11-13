
from ..imaging import PTracer
from ..geometry import Particle
from ..functional import sph2cart, cart2sph

import torch

def D2_several_b_sph(
    particle: Particle,
    bs: torch.Tensor, 
    nsteps = 128, 
    T=10.0,
    D0 = 10.0, 
    dPhi = 0.0, 
    dTh = 0.0,
    trajstyle=None, 
    showic=False,
    save_as=None
    ):
    '''
    Example scenario.

    Works for particles in spherically-symmetric metrics.

    Draws 

    ### Inputs:
    - particle: Particle() - particle to be traced
    - b_s: torch.Tensor() - impact parameters
    - nsteps: int - number of steps (128):
    - T: float - final time (10 by default)
    - D0: float initial distance from the center (10 by default)
    - dPhi: float
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
        trajstyle = ''


    plt.show

    if save_as != None:
        # saving routine
        pass

def flat_axes():



    pass

def flat_plot():


    pass
