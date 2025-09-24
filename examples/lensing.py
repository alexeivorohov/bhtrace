'''
This example produces a detailed lensing picture - i.e. plots lensing function and shows photon motion
'''
import torch
import matplotlib.pyplot as plt
import uniplot as uplt

from bhtrace.geometry import spacetime, Photon
from bhtrace.tracing import PTracer
from bhtrace.scenarios import Lensing
from bhtrace.scenarios.lensing import eval_lens
from bhtrace.functional import LensingPlot


# st = spacetime.KerrSchild(a=0.1)
st = spacetime.SphericallySymmetric()
particle = Photon(st)
tracer = PTracer(eps=1e-5)
tracer.to(dtype=torch.float64)
tracer.__const_dx__ = True

x0 = torch.zeros(2, 4)
x0[..., 1] = 20
x0[..., 2] = torch.tensor([0., 16.])

v0 = torch.Tensor([0., -1.0, 0., 0.])

Lensing._eps_ = 0.1
x, dphi, traj = Lensing.forward(particle, tracer, x0, v0, nsplits=7, T=300, nsteps=128)


fig1 = traj.plot2d()
fig2 = traj.plot_conservation()
# fig3 = traj.plot_metrics()

fig4, ax = plt.subplots(1,1,figsize=(8,8))

ax.plot(x[..., 2], dphi/torch.pi/2)
ax.grid(True)

plt.show()


