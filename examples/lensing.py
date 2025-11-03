'''
This example produces a detailed lensing picture - i.e. plots lensing function and shows photon motion
'''
import torch
import matplotlib.pyplot as plt
import uniplot as uplt

from bhtrace.geometry import spacetime, Photon
from bhtrace.tracing import PTracer
from bhtrace.scenarios import Lensing

from bhtrace.graphics import LensingPlot, Plot2D, PlotValue

# st = spacetime.KerrNewmanBL(a=0.5, q=0.5)
st = spacetime.KerrBL(a=0.1)
particle = Photon(st)
tracer = PTracer(eps=1e-5, ode_method='VCAB4')
tracer.to(dtype=torch.float64)
tracer.__const_dx__ = True

x0, v0, e_b = Lensing.prepare_ic()

Lensing._eps_ = 0.5
x, dphi, traj = Lensing.forward(particle, tracer, x0, v0, nsplits=5, T=60, nsteps=128)

fig = LensingPlot.plot(dphi, x[..., 2])
# figs = traj.report()
fig, _ = PlotValue.plot_hmlt_stat(traj=traj)

plt.show()





