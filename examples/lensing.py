'''
This example produces a detailed lensing picture - i.e. plots lensing function and shows photon motion
'''
import torch
import matplotlib.pyplot as plt

from bhtrace.geometry import spacetime, Photon
from bhtrace.tracing import PTracer
from bhtrace.scenarios import Lensing
from bhtrace.functional import LensingPlot

st = spacetime.KerrSchild(a=0.1)
particle = Photon(st)
tracer = PTracer()

x0 = torch.zeros(2, 4)
x0[..., 1] = 20
x0[..., 2] = torch.tensor([0., 16.])

v0 = torch.Tensor([0., -1.0, 0., 0.])

traj = Lensing.forward(particle, tracer, x0, v0, nsplits=5, T=30, nsteps=128)


