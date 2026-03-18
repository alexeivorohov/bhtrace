import pathlib
from typing import Tuple

import torch
import matplotlib.pyplot as plt

from bhtrace.geometry import Photon, Observer
import bhtrace.geometry.spacetime as st
from bhtrace.tracing import PTracer

import bhtrace.medium as bhM
import bhtrace.geometry.electrodynamics as ed

from bhtrace.grrt.runner import GRRT
from bhtrace.grrt.radiation import IntegralFlux

# --- Configuration ---
N = 128  # Image resolution
D = 32
B0 = 0.5
KERR_A = 0.6
EH_H = 10.0
OBS_R = 20.0
OBS_INCLINATION = torch.pi / 2
TRACE_T = 50
TRACE_NSTEPS = 256
TRACE_R_MAX = 30
TRACE_EPS = 1e-3
MEDIUM_R_IN = 6.0
MEDIUM_R_OUT = 20.0

BASE = st.KerrBL(a=0.6)

def E(X: torch.Tensor) -> torch.Tensor:
    """Electric field function (zero in this case)."""
    return torch.zeros_like(X)

def B(X: torch.Tensor) -> torch.Tensor:
    """Split-monopole magnetic field."""
    r = X[..., 1]
    r2 = torch.pow(r, 2)
    sgn = 1.0
    f = torch.pow(1 + 2 / r, -0.5)
    outp = torch.zeros_like(X)
    outp[..., 1] = B0 / r2 * sgn * f
    return outp

# --- Runner ---

def run_grrt(spacetime: st.Spacetime) -> Tuple[torch.Tensor, bhM.Medium]:
    """
    Sets up and runs the GRRT simulation for a given spacetime.

    Args:
        spacetime: The spacetime geometry to use.

    Returns:
        A tuple containing the computed image flux and the medium used.
    """
    # 1. Ray-tracing setup
    print(f"1. Setting up ray-tracing for {spacetime.__class__.__name__}...")
    photon = Photon(spacetime=spacetime)
    obs = Observer(
        spacetime=spacetime,
        r=OBS_R,
        inclination=OBS_INCLINATION,
        u=torch.Tensor([1, 0, 0, 0]),
    )
    obs.generate_net(net_shape="square", net_rng=(N, N), net_size=(D, D))
    X0, P0 = obs.setup_ic(photon)

    # Determine device
    dev = "cpu"
    print(f"Using device: {dev}")

    # Run tracer
    print("2. Tracing trajectories...")
    tracer = PTracer(ode_method="VCABM4")
    traj = tracer.forward(
        photon, X0, P0, T=TRACE_T, nsteps=TRACE_NSTEPS, r_max=TRACE_R_MAX, eps=TRACE_EPS, device=dev
    )

    # 2. GRRT setup and computation
    print("3. Setting up and running GRRT...")
    medium = bhM.VolumetricShell(BASE, 6.0, 20.0)
    grrt = GRRT(medium=medium, compute_total=True)
    flux_model = IntegralFlux()
    grrt.attach_models(total_models=[flux_model])
    grrt.compute(traj)

    # 3. Retrieve results
    print("4. Retrieving results...")
    total_flux = grrt.retrieve("total")
    image_flux = total_flux.view(N, N).cpu().transpose(-1, -2)

    return image_flux, medium


def main():
    """
    Main function to run the GRRT simulations and generate plots.
    """
    ed_eh = ed.EulerHeisenberg(h=EH_H)

    spacetimes = {
        "kerr": BASE,
        "euler_heisenberg": st.EffGeom(ed_eh, BASE, E, B),
    }

    # 1. Run simulations and store results
    results = {}
    for name, spacetime_obj in spacetimes.items():
        image_flux, medium = run_grrt(spacetime_obj)
        results[name] = (image_flux, medium)

    # 2. Process fluxes and determine global scale
    cleaned_fluxes = {}
    all_flux_tensors = []
    for name, (image_flux, medium) in results.items():
        # Clean up the flux data
        image_flux = torch.nan_to_num(image_flux, torch.nanmean(image_flux))
        a = image_flux.median()
        image_flux[image_flux > 5 * a] = a
        cleaned_fluxes[name] = image_flux
        all_flux_tensors.append(image_flux)

    if not all_flux_tensors:
        print("No fluxes to plot.")
        return

    # Determine global min and max for color scale
    stacked_fluxes = torch.stack(all_flux_tensors)
    vmin = stacked_fluxes.min()
    vmax = stacked_fluxes.max()

    # 3. Plot results with unified scale
    num_spacetimes = len(cleaned_fluxes)
    fig, axes = plt.subplots(1, num_spacetimes, figsize=(10 * num_spacetimes, 10))
    if num_spacetimes == 1:
        axes = [axes]

    for i, (name, image_flux) in enumerate(cleaned_fluxes.items()):
        medium = results[name][1]  # Get medium for title
        ax = axes[i]
        im = ax.imshow(image_flux, cmap="afmhot", vmin=vmin, vmax=vmax)
        ax.set_title(f"Total Flux ({medium.__class__.__name__} on {name})")
        ax.axis("off")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    file_path = pathlib.Path(__file__)
    output_path = file_path.parent / f"{file_path.stem}.png"
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
