"""
This module describes methods for making Trajectory objects in one function call.

"""
from typing import Literal, Optional, Any, Dict

import torch

from bhtrace.data import Trajectory
from bhtrace.geometry import Spacetime, Observer, Particle, PARTICLE_REGISTRY
from bhtrace.geometry.spacetime import SphericallySymmetric, KerrBL
from bhtrace.exact.newtonian import KeplerianTrajectories
import bhtrace.tracing as tracers

ODE_METHOD = "VCABM4"
EPS = 0.001


def make_keplerian(
    net_type: str = 'grid',
    net_params: Optional[Dict[str, Any]] = None,
    obs_params: Optional[Dict[str, Any]] = None,
    nsteps: int = 256,
    T: float = 60,
    particle: str = 'baryon',
    particle_params: Optional[Dict[str, Any]] = None,
    device: str = 'cpu',
) -> 'Trajectory':
    
    raise NotImplementedError('This function is not implemented yet')
    st = SphericallySymmetric() # schwarzschild for now, substitute with parameters for newtonian geometry
    particle = PARTICLE_REGISTRY.create(particle, **particle_params)

    obs = Observer(spacetime=st, **obs_params)
    
    # --- code behind should be wrapped in AnalyticTracer ---

    keplerian = KeplerianTrajectories.from_cartesian(
        x=..., v=...
    )

    timesteps = torch.linspace(0, T, nsteps)
    x, v = keplerian.propagate()

    traj = Trajectory(
        X=...,
        P=...,
        affine_t=timesteps,
        tracer=...,
    )

    return traj

def make_schwarzschild(
    net_type: str = 'square',
    net_params: Optional[Dict[str, Any]] = None,
    obs_params: Optional[Dict[str, Any]] = None,
    nsteps: int = 256,
    T: float = 60,
    particle: str = 'photon',
    particle_params: Optional[Dict[str, Any]] = None,
    r_max: float = 30.0, 
    device: str = 'cpu',
    dtype: str = 'float32',
) -> 'Trajectory':
    
    net_params = net_params or {}
    obs_params = obs_params or {}
    particle_params = particle_params or {}

    st = SphericallySymmetric()
    particle = PARTICLE_REGISTRY.create(particle, spacetime=st, **particle_params)
    
    obs = Observer(st, **obs_params)
    obs.generate_net(net_type)
    x0, p0 = obs.setup_ic(particle)

    tracer = tracers.PTracer(ODE_METHOD, EPS)

    return tracer.forward(
        particle, x0, p0, T=T, nsteps=nsteps, r_max=r_max, device=device, dtype=dtype
    )

    
def make_kerr(
    a: float = 0.6,
    net_type: str = 'square',
    net_params: Optional[Dict[str, Any]] = None,
    obs_params: Optional[Dict[str, Any]] = None,
    nsteps: int = 256,
    T: float = 60,
    particle: str = 'photon',
    particle_params: Optional[Dict[str, Any]] = None,
    r_max: float = 30.0, 
    device: str = 'cpu',
    dtype: str = 'float32',
) -> 'Trajectory':
    
    net_params = net_params or {}
    obs_params = obs_params or {}
    particle_params = particle_params or {}

    st = KerrBL(a=a)
    particle = PARTICLE_REGISTRY.create(particle, spacetime=st, **particle_params)
    
    obs = Observer(st, **obs_params)
    obs.generate_net(net_type)
    x0, p0 = obs.setup_ic(particle)

    tracer = tracers.PTracer(ODE_METHOD, EPS)

    return tracer.forward(
        particle, x0, p0, T=T, nsteps=nsteps, r_max=r_max, device=device, dtype=dtype
    )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    traj = make_kerr()
    fig, ax = traj.plot2d()
    plt.show()
