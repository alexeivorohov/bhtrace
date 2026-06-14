"""
Provides methods for 2D plotting of trajectories.
"""

from typing import Dict, Tuple, Iterable, List, Optional, Any, Literal, Protocol, runtime_checkable

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.cm import ScalarMappable

from bhtrace.utils.registry import CallableRegistry
from bhtrace.graphics.utils import figure_handler, add_info_text, _set_borders, _normalize_trajectories
from bhtrace.graphics.coloring import _multicolored_lines_2d, _get_scalar_mappable, _normalize_rgba_to_list
from bhtrace.graphics.horizons import horizon_patch_2d
from bhtrace.graphics.scaling import scale


class Traj2DBackend(Protocol):

    def __call__(
        xy: np.ndarray,
        rgba: Optional[np.ndarray],
        label: Optional[str],
        ax: Optional[plt.Axes],
        fig: Optional[plt.Figure],
        **kwargs,
    ):
        """
        Backends for 2d trajectory plot with horizon and coloring
 
        Parameters
        ----------
        x : np.ndarray | list of np.ndarray
            first coordinate of trajectory projection, shape (..., steps, 2)
        rgba : optional, np.ndarray
            Values for coloring the trajectories, of shape `(..., steps)` or
            `(..., 1)`. If provided, `color` should be a colormap.
        label : str, optional
            Label to add on a plot legend for this trajectory
        info_text : str, optional
            Additional text to display at the bottom of the figure.
        ax : matplotlib.axes.Axes, optional
            A matplotlib axes object to plot on. If `None`, a new one is created.
        fig : matplotlib.figure.Figure, optional
            A matplotlib figure object. If `None`, a new one is created.
        **kwargs
            Additional keyword arguments are passed to the underlying plotting
            function
        """
        pass


TRAJ_2D_BACKEND_REGISTRY = CallableRegistry(Traj2DBackend)


@TRAJ_2D_BACKEND_REGISTRY.register('mpl', 'matplotlib')
def _mpl_traj_2d_plotter(
    xy: np.ndarray,
    rgba: Optional[np.ndarray],
    ax: Optional[plt.Axes],
    fig: Optional[plt.Figure],
    label: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot one batch of particle trajectories on a 2D plane.

    This is a low-level plotting function.

    Parameters
    ----------
    coords : numpy.ndarray or list of numpy.ndarray
        Trajectory coordinates. Can be a single trajectory (N, D), a batch (B, N, D),
        or a list of trajectories (ragged batch).
    q : torch.Tensor, optional
        Values for coloring the trajectories, shape [batch, steps] or [batch, 1].
        If provided, trajectories are colored using a colormap.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object to plot on. If None, a new one is created.
    fig : matplotlib.figure.Figure, optional
        A matplotlib figure object. If None, a new one is created.
    **kwargs
        Additional keyword arguments are passed to `matplotlib.axes.Axes.plot`
        or `_multicolored_lines_2d`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    """
    fig, ax = figure_handler(fig, ax, figsize=(8, 8))

    if rgba is not None: #or isinstance(xy)
        
        xy_list = _normalize_trajectories(xy)
        rgba_list = _normalize_rgba_to_list(rgba, xy_list)
        _multicolored_lines_2d(
            xy_list=xy_list, rgba_list=rgba_list, ax=ax, **kwargs,
        )

    else:
        print("Called normal plot")
        ax.plot(
            xy[..., 0].T,
            xy[..., 1].T,
            label=label,
            # color=color,
            **kwargs,
        )

    return fig, ax


def plot2d(
    xy: np.ndarray,
    q: np.ndarray = None,
    borders: Optional[Tuple[float, float, float, float]] = None,
    horizon: float | np.ndarray = None,
    horizon_param: Dict[str, Any] = None,
    backend: str = "mpl",
    cmap: str | Colormap = None,
    sm: ScalarMappable = None,
    q_scale: str = 'linear',
    q_label: str = None,
    label: str = None,
    info_text: str = None,
    ax: plt.Axes = None,
    fig: plt.Figure = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D projection of trajectory data

    Parameters
    ----------
    xy : np.ndarray
        Trajectory coordinates of shape `(..., steps, 2)`
    q : np.ndarray, optional
        Scalar values for coloring the trajectories, of shape `(..., steps)` or
        `(..., 1)`. If provided, `color` should be a colormap.
    borders : tuple, optional
        A tuple (xmin, xmax, ymin, ymax) to set the axes limits. If not provided,
        limits are determined automatically from the data.
    q_label : str, optional
        Label for the colorbar if `q` is provided.
    color : str or matplotlib.cm.ScalarMappable, optional
        Color specification for the trajectory. If a string, it should be a
        valid matplotlib color. If `q` is specified, this should be a
        `ScalarMappable` to map `q` values to colors.
    label : str, optional
        Label for the trajectory, to be shown in a legend.
    info_text : str, optional
        Additional text to display at the bottom of the figure.
    horizon : float or numpy.ndarray, optional
        If provided, plots a horizon as a circle with this radius.
    horizon_param : dict, optional
        Additional parameters for the horizon patch (e.g., `color`, `alpha`).
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object to plot on. If `None`, a new one is created.
    fig : matplotlib.figure.Figure, optional
        A matplotlib figure object. If `None`, a new one is created.
    **kwargs
        Additional keyword arguments are passed to the underlying plotting
        function (`_plot_traj`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    """
    fig, ax = figure_handler(fig, ax)

    

    if q is not None and sm is None:
        if q_scale != 'linear':
            q = scale(q, q_scale)
        if cmap is None:
            raise ValueError("A colormap 'cmap' must be provided to color the trajectory.")
        sm = _get_scalar_mappable([q], cmap=cmap)
        rgba = sm.to_rgba(q)
    else:
        rgba = None

    _set_borders(ax, borders=borders, coords=xy)


    if horizon is not None:
        horizon_param = horizon_param or {}
        horizon_patch = horizon_patch_2d(horizon, **horizon_param)
        ax.add_patch(horizon_patch)
    
    if sm is not None:
        fig.colorbar(sm, ax=ax, label=q_label)

    if info_text:
        add_info_text(fig, info_text)
 
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    ax.grid(True)
    if label:
        ax.legend()

    outp = TRAJ_2D_BACKEND_REGISTRY[backend](
        xy=xy,
        rgba=rgba,
        ax=ax,
        fig=fig,
        **kwargs,
    )

    return outp