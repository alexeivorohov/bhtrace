"""
This file contains functions for generating human-readable reports from GRRT computations.
"""
from typing import TYPE_CHECKING
import torch
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from bhtrace.grrt.runner import GRRT
    from bhtrace.data import Trajectory

def tracing_report(trajectory: 'Trajectory'):
    ...

def grrt_report(grrt: 'GRRT', trajectory: 'Trajectory', image_shape: tuple = None):
    """
    Generates a report with plots for a GRRT computation.

    Parameters
    ----------
    grrt : GRRT
        The GRRT object after running compute with a probe_idx.
    trajectory : Trajectory
        The trajectory object.
    image_shape : tuple, optional
        The shape of the output image, if the trajectory is a grid. Defaults to None.
    """
    history = grrt.probe_history
    if not history:
        print("No probe history found. Run GRRT.compute with a probe_idx set.")
        return

    # --- Plotting ---
    num_steps = len(history.x)
    steps = np.arange(num_steps)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Plot coordinate components
    x_coords = torch.stack(history.x)
    labels = ['t', 'r', 'theta', 'phi']
    for i in range(4):
        axs[0].plot(steps, x_coords[:, i], label=labels[i])
    axs[0].set_title(f'Coordinate Components for Particle {grrt.probe_idx}')
    axs[0].set_ylabel('Coordinate Value')
    axs[0].legend()
    axs[0].grid(True)
    
    # Mark hits
    is_hit = torch.tensor(history.is_hit)
    hit_steps = (is_hit).nonzero(as_tuple=True)[0]
    for step in hit_steps:
        axs[0].axvline(step, color='r', linestyle='--', alpha=0.5)

    # Plot redshift
    z_values = np.array(history.z)
    axs[1].plot(steps, z_values, 'o-')
    axs[1].set_title(f'Redshift (z) for Particle {grrt.probe_idx}')
    axs[1].set_ylabel('Redshift (z)')
    axs[1].grid(True)

    # Plot hit status
    axs[2].plot(steps, is_hit.float(), 'o-')
    axs[2].set_title(f'Hit Status for Particle {grrt.probe_idx}')
    axs[2].set_ylabel('Hit (1) or Miss (0)')
    axs[2].set_xlabel('Time Step')
    axs[2].grid(True)
    
    fig.tight_layout()
    
    # Final image
    if image_shape:
        if grrt.spectrum is not None and grrt.spectrum.numel() > 0:
            # Assuming spectrum is (num_rays, num_freqs)
            num_freqs = grrt.spectrum.shape[-1]
            image_data = grrt.spectrum.view(*image_shape, num_freqs).detach().cpu().numpy()
            # Plot the first frequency
            plt.figure(figsize=(8, 8))
            plt.imshow(image_data[..., 0].squeeze(), origin='lower')
            plt.title('Final Image (Spectrum)')
            plt.colorbar()
        elif grrt.total_flux is not None and grrt.total_flux.numel() > 0:
            image_data = grrt.total_flux.view(*image_shape).detach().cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(image_data.squeeze(), origin='lower')
            plt.title('Final Image (Total Flux)')
            plt.colorbar()

    return fig
