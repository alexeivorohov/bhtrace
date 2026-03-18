"""


"""

from typing import Dict, Tuple, Iterable, TYPE_CHECKING

import torch
import matplotlib.pyplot as plt
import math
import numpy as np

if TYPE_CHECKING:
    from bhtrace.trajectory.trajectory import Trajectory

class PlotValue:
    """
    A collection of methods for plotting various quantities from a Trajectory object.
    """

    @classmethod
    def plot(
        cls,
        traj: 'Trajectory',
        value_func, 
        name='Q', 
        labels=None, 
        mask=None, 
        fig: plt.Figure = None, 
        ax: plt.Axes = None
    ):
        """
        Plots a custom quantity derived from the trajectory.
        The function `value_func` should take the trajectory object and return a tensor of shape (ntraj, nsteps, ...).
        """
        if mask is None:
            mask = torch.ones(traj.ntraj, dtype=torch.bool)

        quantity = value_func(traj)
        print(f'Shape: {quantity.shape}')
        quantity = quantity[mask, ...].detach().cpu()

        if quantity.ndim > 2:
            flat_quantity = quantity.reshape(*quantity.shape[:2], -1)
        else:
            flat_quantity = quantity.unsqueeze(-1)

        n_components = flat_quantity.shape[2]

        if labels is None:
            labels = [f'{name}_{i}' for i in range(n_components)] if n_components > 1 else [name]
        
        if len(labels) != n_components:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of components ({n_components})")

        components = [flat_quantity[:, :, i] for i in range(n_components)]

        # Single component case
        if n_components == 1:
            if ax is None:
                if fig is None:
                    fig, ax = plt.subplots(figsize=(10, 5))
                else:
                    ax = fig.add_subplot(111)
            else:
                fig = ax.get_figure()
            
            ax.plot(components[0].numpy().T)
            ax.set_title(f'{labels[0]} along trajectory')
            ax.set_ylabel(labels[0])
            ax.grid(True)
            ax.set_xlabel('time step')
            if fig:
                fig.tight_layout()
            return fig

        # Multiple components case
        ncols = 2
        nrows = math.ceil(n_components / ncols)
        figsize = (15, 5 * nrows)

        if fig is None:
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)
        else:
            axs = fig.subplots(nrows, ncols, squeeze=False, sharex=True)
        axs = axs.flatten()

        for i, label in enumerate(labels):
            ax_ = axs[i]
            ax_.plot(components[i].numpy().T)
            ax_.set_title(f'{label} along trajectory')
            ax_.set_ylabel(f'{label}')
            ax_.grid(True)
        
        for ax_ in axs.reshape(-1, ncols)[-1, :]:
            ax_.set_xlabel('time step')

        for i in range(n_components, len(axs)):
            axs[i].set_visible(False)

        fig.tight_layout()
        return fig

    @classmethod
    def plot_energy_deviation(cls, traj: 'Trajectory', ax: plt.Axes = None):
        """
        Plots the deviation of the Hamiltonian from its initial value along the trajectory.
        """
        machine_eps = torch.finfo(traj.X.dtype).eps

        def value_func(t: 'Trajectory'):
            return torch.log10(torch.abs(t.particle.hmlt(t.X, t.P) - t.particle.mu))

        return cls.plot(traj, value_func, title=f'$\\log_{(10)} |H - \\mu|, machine epsilon={machine_eps:.2e}$', ax=ax)
    
    @classmethod
    def plot_energy_deviation_histogram(cls, traj: 'Trajectory', ax: plt.Axes = None):
        """
        Plots a histogram of the deviation of the Hamiltonian from its initial value along the trajectory.
        """
        machine_eps = torch.finfo(traj.X.dtype).eps

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(8,8))
        else:
            fig = ax.get_figure()

        dH = torch.log10(torch.abs(traj.particle.hmlt(traj.X, traj.P) - traj.particle.mu))

        bins = np.arange(-8, 2, 1)
        ax.hist(x=dH.flatten(), bins=bins, density=True)
        ax.grid(True)
        ax.set_xlabel(f'$\\log_{10} |H - \\mu|$')
        ax.set_ylabel('Density')
        ax.axvline(x=np.log10(machine_eps), color='red', linestyle='--', label=f'machine epsilon={machine_eps:.2e}')
        ax.legend()
        return fig, ax
    
    @classmethod
    def plot_metrics(cls, traj: 'Trajectory', ax: plt.Axes = None):

        def value_func(t: 'Trajectory'):
            return t.spacetime.g(t.X)
        
        return cls.plot(traj, value_func, ax=ax)