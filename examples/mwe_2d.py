import pathlib

import torch
import matplotlib.pyplot as plt
import numpy as np

from bhtrace.geometry import Photon, Observer
from bhtrace.geometry.spacetime import KerrBL
from bhtrace.tracing import PTracer

spacetime = KerrBL(a=0.0)
photon = Photon(spacetime=spacetime)

obs = Observer(
    spacetime=spacetime,
    position=torch.Tensor([0, 20, 0, 0]),
    camera_dir=torch.Tensor([-1, 0, 0]),
    u = torch.Tensor([1, 0, 0, 0])
)

N = 64

obs.generate_net(
    net_shape='square',
    net_rng = (N, 1),
    net_size = (32, 0)
)

X0, P0 = obs.setup_ic(photon)

tracer = PTracer(ode_method='VCAB4')
tracer.__const_dx__ = False
traj = tracer.forward(photon, X0, P0, T=60, nsteps=256, r_max=30, max_proper_t = 500, eps=1e-3, device='cuda')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('2D Trajectories and Conservation')

traj.plot2d(ax=ax1, projection='yz')
traj.histogram(ax=ax2, replace_nan=-1.0, cleaned=False, q_scale='log', density=False, bins=20)

plt.tight_layout()

plt.show()
file_path = pathlib.Path(__file__)
output_path = file_path.parent / f"{file_path.stem}.png"
print(f"Saving plot to {output_path}")
plt.savefig(output_path)
plt.close(fig)