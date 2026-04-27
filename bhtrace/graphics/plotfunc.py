"""
Plotting functions for various quantities.
"""
from typing import List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
import math
import numpy as np

from bhtrace.graphics.utils import figure_handler, add_info_text


def plot(
    quantity: torch.Tensor,
    name: str = 'Q',
    labels: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    info_text: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a quantity along a trajectory or against a parameter.

    The quantity tensor is expected to be of shape (ntraj, nsteps, ...),
    where ntraj is the number of trajectories/rays, and nsteps is the number of
    time/parameter steps. This function can handle multi-component quantities
    by plotting each component on a separate subplot.

    Parameters
    ----------
    quantity (torch.Tensor): 
        The tensor to plot. Shape (ntraj, nsteps, ...).
    name (str, optional): The base name for the quantity. Defaults to 'Q'.
    labels (List[str], optional): A list of labels for each component of the quantity.
                                    If None, labels are generated automatically. Defaults to None.
    fig (plt.Figure, optional): A matplotlib figure object. If None, creates a new one.
    ax (plt.Axes, optional): A matplotlib axes object. If None, creates a new one.
                                Only used if the quantity has a single component.
    info_text (str, optional): Additional text to display at the bottom of the figure.
    **kwargs: Additional keyword arguments passed to ax.plot().

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]: 
        The figure and axes (or dictionary of axes) of the plot.
    """
    quantity = quantity.detach().cpu()

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
        fig, ax = figure_handler(fig, ax, figsize=(10, 5))
        ax.plot(components[0].numpy().T, **kwargs)
        ax.set_title(f'{labels[0]} along trajectory')
        ax.set_ylabel(labels[0])
        ax.grid(True)
        ax.set_xlabel('time step')
        if info_text:
            add_info_text(fig, info_text)
        fig.tight_layout()
        return fig, ax

    # Multiple components case
    ncols = 2
    nrows = math.ceil(n_components / ncols)
    figsize = (15, 5 * nrows)

    if fig is None:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True)
    else:
        # This will clear the figure, which might not be desired.
        # A better approach might be to expect a grid of axes.
        # For now, we'll let it create new subplots on the existing figure.
        axs = fig.subplots(nrows, ncols, squeeze=False, sharex=True)
    
    axs = axs.flatten()

    for i, label in enumerate(labels):
        ax_ = axs[i]
        ax_.plot(components[i].numpy().T, **kwargs)
        ax_.set_title(f'{label} along trajectory')
        ax_.set_ylabel(f'{label}')
        ax_.grid(True)

    for ax_ in axs.reshape(-1, ncols)[-1, :]:
        ax_.set_xlabel('time step')

    for i in range(n_components, len(axs)):
        axs[i].set_visible(False)
    
    if info_text:
        add_info_text(fig, info_text)
        
    fig.tight_layout()
    return fig, axs


