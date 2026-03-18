import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from bhtrace.geometry import spacetime, Photon, Observer
import bhtrace.geometry.electrodynamics as ed
from bhtrace.tracing import PTracer
from bhtrace.scenarios import Lensing
from bhtrace.graphics import LensingPlot, Plot2D, PlotValue

# --- Configuration ---
N = 32 # Number of initial rays
B_SPAN = 20
B0 = 0.5
OBS_R = 20.0
OBS_INCLINATION = torch.pi / 2
KERR_A = 0.6
EH_H = 10.0
TRACE_T = 50
TRACE_NSTEPS = 512
TRACE_EPS = 1e-4
ODE_METHOD = 'VCABM4'
N_SPLITS = 4

ED_EH = ed.EulerHeisenberg(h=EH_H)
ED_M = ed.Maxwell()

def E(X):
    return torch.zeros_like(X)

def B(X):
    r = X[..., 1]
    r2 = torch.pow(r, 2)

    sgn = 1.0 #torch.sign(X[..., 3]).unsqueeze(-1)
    f = torch.pow(1+2/r, -0.5)
    
    outp = torch.zeros_like(X)
    outp[..., 1] = B0/r2*sgn*f
    return outp

background = spacetime.KerrBL(a=KERR_A)

SPACETIMES = {
    'base': background,
    'eh_eff': spacetime.EffGeom(ED_EH, background, E, B),
}

# --- Run simulations ---

outp = {}

for name, st in SPACETIMES.items():

    particle = Photon(st)
    tracer = PTracer(eps=TRACE_EPS, ode_method=ODE_METHOD)
    tracer.to(dtype=torch.float32, dev='cuda')
    tracer.__const_dx__ = True

    observer = Observer(st, r=OBS_R, inclination=OBS_INCLINATION)
    observer.generate_net('line', net_rng=(N,), net_size=(B_SPAN, 0))

    lensing = Lensing(tracer, observer, particle)

    x, dphi, traj = lensing.forward(nsplits=N_SPLITS, T=TRACE_T, nsteps=TRACE_NSTEPS)

    outp[name] = x, dphi, traj

dphi_dict = {k: v[1] for k, v in outp.items()}
b_dict = {k: v[0][..., 2] for k, v in outp.items()}
traj_dict = {k: v[2] for k, v in outp.items()}

# --- Plotting ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Lensing: comaprison and photon energy deviations for Kerr BH and Kerr BH in Euler-Heisenberg effective geometry')
dataset_colors = plt.cm.viridis(np.linspace(0, 1, len(dphi_dict)))

ax1.set_title("Lensing Curves")
for i, key in enumerate(dphi_dict.keys()):
    b = b_dict[key]
    dphi = dphi_dict[key]
    sorted_indices = torch.argsort(b)
    LensingPlot.plot1(b[sorted_indices], dphi[sorted_indices], ax1, dataset_color=dataset_colors[i], label=key)
ax1.set_xlabel("Impact parameter b")
ax1.set_ylabel("Number of windings n")
ax1.grid(True)
ax1.legend()

ax2.set_title("Hamiltonian Conservation")
for i, key in enumerate(traj_dict.keys()):
    traj = traj_dict[key]
    traj.plot_energy_deviation_histogram(ax2)
ax2.legend()

plt.tight_layout()
file_path = pathlib.Path(__file__)
output_path = file_path.parent / f"{file_path.stem}.png"
print(f"Saving plot to {output_path}")
plt.savefig(output_path)
plt.close(fig)
