import os

import torch

from bhtrace.geometry import Spacetime, SphericallySymmetric, Photon, Particle, Observer
from bhtrace.tracing import PTracer
from bhtrace import Trajectory

directory = os.path.dirname(os.path.abspath(__file__))
pathname = '/data/mwe_sph_2d'
formats = ['.png']
file_path = directory + pathname

if not os.path.exists(file_path + '.traj'):

    spacetime = SphericallySymmetric()
    photon = Particle(name='Photon', spacetime=spacetime)

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
        net_size = (64, 0)
        )

    tracer = PTracer(ode_method='RK4')
    traj = tracer.forward(photon, X0, P0, T=30, nsteps=128, r_max=30, max_proper_t = 500, eps=1e-3)
    traj.save(file_path + '.traj')
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
