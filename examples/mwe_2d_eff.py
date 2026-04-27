import pathlib
import torch
import matplotlib.pyplot as plt

from bhtrace.geometry.spacetime import KerrBL, EffGeom
from bhtrace.geometry.particle import Photon
from bhtrace.geometry.electrodynamics import EulerHeisenberg
from bhtrace.geometry import Observer
from bhtrace.tracing import PTracer
from bhtrace import Trajectory

ED = EulerHeisenberg(h=1)
def E(X):
    return torch.zeros_like(X)

def B(X):
    B0 = 0.1
    r2 = (X[..., 1]**2 + X[..., 2]**2 + X[..., 3]).unsqueeze(-1)
    r = torch.pow(r2, 0.5)
    e_r = X[..., 1:]/r

    sgn = torch.sign(X[..., 3]).unsqueeze(-1)
    B_r = B0/r2*sgn*torch.pow(1+2/r, -0.5)

    outp = torch.zeros_like(X)
    outp[..., 1:] = B_r*e_r
    return outp

background = KerrBL(a=0.6)
spacetime = EffGeom(ED, background, E, B)

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

X0, P0 = obs.setup_ic(
    photon,
    )

if torch.any(torch.isnan(P0)):
    print(P0)

tracer = PTracer(ode_method='RK4')
traj = tracer.forward(photon, X0, P0, T=30, nsteps=128, r_max=30, max_proper_t = 500, eps=1e-3)

# --- Plotting ---
fig, axes = plt.subplots(3, 2, figsize=(20, 30))
fig.suptitle('MWE 2D Effective Geometry Analysis', fontsize=16)
axes = axes.flatten()

# Plot 2D trajectory
traj.plot2d(ax=axes[0])
axes[0].set_title('2D Trajectory')

# Plot Hamiltonian conservation
traj.plot_conservation(ax=axes[1])
axes[1].set_title('Hamiltonian Conservation')

# Plot impulses (assuming it takes an ax)
try:
    traj.plot_impulses(ax=axes[2])
    axes[2].set_title('Impulses')
except (AttributeError, TypeError):
    axes[2].text(0.5, 0.5, 'plot_impulses not available or failed', ha='center', va='center')
    axes[2].set_title('Impulses')

# Plot coordinates (assuming it takes an ax)
try:
    traj.plot_coords(ax=axes[3])
    axes[3].set_title('Coordinates')
except (AttributeError, TypeError):
    axes[3].text(0.5, 0.5, 'plot_coords not available or failed', ha='center', va='center')
    axes[3].set_title('Coordinates')

# Plot metrics (assuming it takes an ax)
try:
    traj.plot_metrics(ax=axes[4])
    axes[4].set_title('Metrics')
except (AttributeError, TypeError):
    axes[4].text(0.5, 0.5, 'plot_metrics not available or failed', ha='center', va='center')
    axes[4].set_title('Metrics')

# Plot magnetic field (assuming it takes an ax)
try:
    traj.plot_quantity(ED.B, name='B', ax=axes[5])
    axes[5].set_title('Magnetic Field')
except (AttributeError, TypeError):
    axes[5].text(0.5, 0.5, 'plot_quantity not available or failed', ha='center', va='center')
    axes[5].set_title('Magnetic Field')

plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# Save the figure
file_path = pathlib.Path(__file__)
output_path = file_path.parent / f"{file_path.stem}.png"
print(f"Saving plot to {output_path}")
plt.savefig(output_path)
plt.close(fig)

