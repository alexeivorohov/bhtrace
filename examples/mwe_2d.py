import os

import torch

from bhtrace.geometry import KerrSchild, Photon, Observer
from bhtrace.tracing import PTracer
from bhtrace import Trajectory

directory = os.path.dirname(os.path.abspath(__file__))
file_name = '/data/mwe_2d.traj'
file_path = directory + file_name

if not os.path.exists(file_path):

    spacetime = KerrSchild(a=0.1)
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
        net_size = (64, 0)
        )

    tracer = PTracer(ode_method='RK4')
    # tracer.__const_dx__ = True
    traj = tracer.forward(photon, X0, P0, T=30, nsteps=128, r_max=30, max_proper_t = 500, eps=1e-3)
    traj.save(file_path)
else:
    traj = Trajectory.load(file_path)

fig = traj.plot2d()
fig.show()
fig.savefig(directory + '/data/mwe_2d.png', dpi=300)
# fig.savefig(directory + '/data/mwe_2d.pdf')

fig2 = traj.plot_conservation()
fig2.savefig(directory + '/data/mwe_2d_conservation.png', dpi=300)
# fig2.savefig(directory + '/data/mwe_2d_conservation.pdf')

fig3 = traj.plot_impulses()
fig3.savefig(directory + '/data/mwe_2d_impulses.png', dpi=300)
# fig3.savefig(directory + '/data/mwe_2d_impulses.pdf')

fig4 = traj.plot_coords()
fig4.savefig(directory + '/data/mwe_2d_coords.png', dpi=300)
# fig4.savefig(directory + '/data/mwe_2d_coords.pdf')