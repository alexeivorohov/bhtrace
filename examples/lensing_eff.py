'''
This example produces a detailed lensing picture - i.e. plots lensing function and shows photon motion
'''
import torch
import matplotlib.pyplot as plt
import uniplot as uplt

from bhtrace.geometry import spacetime, Photon
from bhtrace.geometry.electrodynamics import EulerHeisenberg
from bhtrace.tracing import PTracer
from bhtrace.scenarios import Lensing

from bhtrace.graphics import LensingPlot, Plot2D, PlotValue

ED = EulerHeisenberg(h=1)
def E(X):

    return torch.zeros_like(X)

def B(X):
    B0 = 0.1
    # r2 = (X[..., 1]**2 + X[..., 2]**2 + X[..., 3]).unsqueeze(-1)
    r = X[..., 1]
    r2 = torch.pow(r, 2)

    sgn = 1.0 #torch.sign(X[..., 3]).unsqueeze(-1)
    f = torch.pow(1+2/r, -0.5)
    
    outp = torch.zeros_like(X)
    outp[..., 1] = B0/r2*sgn*f
    return outp

background = spacetime.KerrBL(a=0.1)
st = spacetime.EffGeom(ED, background, E, B)

particle = Photon(st)
tracer = PTracer(eps=1e-5)
tracer.to(dtype=torch.float64)
tracer.__const_dx__ = True

x0 = torch.zeros(3, 4)
x0[..., 1] = 20
x0[..., 2] = torch.tensor([-16., 0., 16.])

v0 = torch.Tensor([0., -1.0, 0., 0.])

Lensing._eps_ = 0.5
x, dphi, traj = Lensing.forward(particle, tracer, x0, v0, nsplits=8, T=60, nsteps=128)

PlotValue.plot_conservation(traj)
LensingPlot.plot(dphi, x[..., 2])




