from ..imaging import PTracer
from ..geometry import Particle
from ..functional import sph2cart, cart2sph
from ..radiation import Radiation

import torch

def ImagePix(
    particle: Particle,
    rad_field: Radiation,
    p_width = 256,
    p_height = 256,
    pixel_d = 0.1,
    nsteps = 128, 
    diff_eps = 5e-4,
    T=10.0,
    D0 = 10.0, 
    dPhi = 0.0, 
    dTh = 0.0,
    ode_method = 'Euler',
    save_as = None
    ):

    db_y = p_width*pixel_d/2
    db_z = p_height*pixel_d/2
    y_s = torch.linspace(-db_y, db_y, p_width)
    z_s = torch.linspace(-db_z, db_z, p_height)
    y0, z0 = torch.meshgrid(y_s, z_s, indexing='ij')

    X0 = torch.stack([torch.zeros_like(y0), torch.ones_like(y0)*D0, y0, z0], dim=-1).flatten(start_dim = 0, end_dim=1)

    print(X0.shape)

    p_s0 = torch.ones_like(y0)
    p_s1 = torch.zeros_like(y0)
    DirV = torch.stack([-p_s0, p_s1, p_s1], dim=-1).flatten(start_dim = 0, end_dim=1)

    P0 = torch.stack([particle.GetNullMomentum(X0[i], DirV[i]) for i in range(X0.shape[0])], dim = 0)

    assert(P0.shape == X0.shape, 'X0 and P0 shapes do not match')

    tracer = PTracer(r_max=D0*1.5, method=ode_method)

    X_res, P_res = tracer.trace(X0=X0, P0=P0, eps=diff_eps, nsteps=nsteps)

    X_cart, P_cart = sph2cart(X_res, P_res)


    pass

def ImageSave(path):

    pass

def ImageLoad(path):

    pass