import os

import torch

from bhtrace.geometry.spacetime import SphericallySymmetric
from bhtrace.geometry.particle import Photon
from bhtrace.geometry import Observer
from bhtrace.tracing import PTracer
from bhtrace import Trajectory

directory = os.path.dirname(os.path.abspath(__file__))
pathname = '/data/mwe_sph_2d'
formats = ['.png']
file_path = directory + pathname


spacetime = SphericallySymmetric()
photon = Photon(name='Photon', spacetime=spacetime)
tracer = PTracer(ode_method='RK4')
tracer.__const_dx__ = True

if not os.path.exists(file_path + '.traj'):


    obs = Observer(
        spacetime=spacetime,
        position=torch.Tensor([0, 20, 0, 0]),
        camera_dir=torch.Tensor([-1, 0, 0]),
        u = torch.Tensor([1, 0, 0, 0])
        )

    obs.generate_net(
        net_shape='square',
        net_rng = (64,1),
        net_size = (64, 0)
        )
    X0, P0 = obs.setup_ic(photon)

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

# from bhtrace.geometry.electrodynamics import EulerHeisenberg

# ED = EulerHeisenberg(h=1)
# def E(X):

#     return torch.zeros_like(X)

# def B(X):
#     B0 = 1.0
#     sgn = 1.0 # torch.sign(X[..., 2]-torch.pi/2)
#     sgn = torch.sign(X[..., 3])
#     R2 = X[..., 1]**2 + X[..., 2]**2 + X[..., 3]
#     B_r = B0/R2*sgn*torch.pow(1+2/X[..., 2], -0.5)

#     outp = torch.zeros_like(X)
#     outp[..., 1] = B_r
#     return outp


# fig6 = traj.plot_quantity(B, name='B')
# fig6.savefig(file_path + '_B' + formats[0])

# from bhtrace.scenarios.lensing import Lensing, eval_lens
import matplotlib.pyplot as plt

# x0 = torch.zeros(2, 4)
# x0[..., 1] = 20
# x0[..., 2] = torch.tensor([0., 16.])

# v0 = torch.Tensor([0., -1.0, 0., 0.])

# x, a, traj = Lensing.forward(photon, tracer, x0, v0, nsplits=3, T=50, nsteps=128)

# fig2 = traj.plot2d()
# fig3 = traj.plot_conservation()
# # fig3 = traj.plot_metrics()
# fig4, ax  = plt.subplots(1,1,figsize=(8,8))
# a = eval_lens(traj)
# ax.plot(traj['Cartesian'][0][..., 0, 2], a)

plt.show()

