import os

import torch

from bhtrace.geometry.spacetime import SphericallySymmetric, EffGeom
from bhtrace.geometry.particle import Photon
from bhtrace.geometry.electrodynamics import EulerHeisenberg
from bhtrace.geometry import Observer
from bhtrace.tracing import PTracer
from bhtrace import Trajectory

directory = os.path.dirname(os.path.abspath(__file__))
pathname = '/data/mwe_2d_eff'
formats = ['.png']
file_path = directory + pathname

if not os.path.exists(file_path + '.traj'):

    ED = EulerHeisenberg(h=1)
    def E(X):

        return torch.zeros_like(X)
    
    def B(X):
        B0 = 1.0
        sgn = 1.0 # torch.sign(X[..., 2]-torch.pi/2)
        B_r = B0*torch.pow(X[..., 2], -2)*sgn*torch.pow(1+2/X[..., 2], -0.5)

        outp = torch.zeros_like(X)
        outp[..., 1] = B_r
        return outp

    background = SphericallySymmetric()
    spacetime = EffGeom(ED, background, E, B)
    photon = Photon(spacetime=spacetime)

    obs = Observer(
        spacetime=spacetime,
        position=torch.Tensor([0, 20, 0, 0]),
        camera_dir=torch.Tensor([-1, 0, 0]),
        u = torch.Tensor([1, 0, 0, 0])
        )

    X0, P0 = obs.setup_ic(
        photon,
        net_shape='square',
        net_rng = (64,1),
        net_size = (32, 0)
        )
    
    if torch.any(torch.isnan(P0)):
        print(P0)

    tracer = PTracer(ode_method='RK4')
    traj = tracer.forward(photon, X0, P0, T=30, nsteps=128, r_max=30, max_proper_t = 500, eps=1e-3)
    # traj.save(file_path + '.traj')
else:
    traj = Trajectory.load(file_path + '.traj')

fig = traj.plot2d()
fig.savefig(file_path + formats[0])

fig2 = traj.plot_conservation()
fig2.savefig(file_path + '_conservation' + formats[0])

fig3 = traj.plot_impulses()
fig3.savefig(file_path + '_impulses' + formats[0])

fig4 = traj.plot_coords()
fig4.savefig(file_path + '_coords' + formats[0])

fig5 = traj.plot_metrics()
fig5.savefig(file_path + '_metrics' + formats[0])
