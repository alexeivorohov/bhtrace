"""
Provides methods for 2D plotting of trajectories.
"""

from typing import Dict, Tuple, Iterable, List, Optional, Any, Literal, Protocol

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.cm import ScalarMappable

from bhtrace.utils.registry import CallableRegistry
from bhtrace.graphics.utils import opt_mosaic, figure_handler, add_info_text, Projector
from bhtrace.graphics.coloring import _multicolored_lines_2d, _get_scalar_mappable
from bhtrace.graphics.horizons import horizon_patch_2d

# types of plots:
# 1. 2D trajectory woth optional coloring - <- FOCUS
# 2. 2D trajectory mosaic - reuses previous
# 3. 2D trajectory stacked

# sole responsibility
# plot2d - can plot exactly one batch of trajectories with/without horizons
# plotting pipeline:
# 1. handle input data in a standartized way:
# - projection (independent of backend)
# - colors (independent of backend? mpl supports RGBA and RGB, uniplot supports RGB(int), HEX)
# - colormaps????
# - horizon
# - ...
# 2. pass data to the plotter 
# 3. apply backend-specific tricks, if necessary? (may be )
# 
# how to call horizon plotter in uniplot independently?
# no way - unless some kind of wraparound is done. 

# handle figure and axes
# plot trajectories
# plot horizon (if specified )

class Traj2DBackend(Protocol):

    def __call__(
        x: np.ndarray,
        y: np.ndarray,
        q: Optional[np.ndarray],
        borders: Optional[Tuple[float, float, float, float]],
        horizon: float | np.ndarray,
        horizon_param: Dict[str, Any],
        projection: str | np.ndarray,
        cmap: str | Colormap,
        sm: ScalarMappable,
        q_label: Optional[str],
        label: Optional[str],
        info_text: Optional[str],
        ax: Optional[plt.Axes],
        fig: Optional[plt.Figure],
        **kwargs,
    ):
        """
        Backends for 2d trajectory plot with horizon and coloring
 
        Parameters
        ----------
        x : np.ndarray
            first coordinate of trajectory projection, shape (..., steps)
        y : np.ndarray
            second coordinate of trajectory projection, shape (..., steps)
        q : optional, np.ndarray
            Values for coloring the trajectories, of shape `(..., steps)` or
            `(..., 1)`. If provided, `color` should be a colormap.
        borders : tuple, optional
            A tuple (xmin, xmax, ymin, ymax) to set the axes limits.
        q_label : str, optional
            Label for the colorbar if `q` is provided.
        color : str or matplotlib.cm.ScalarMappable, optional
            Color specification for the trajectory. If a string, it should be a
            valid matplotlib color. If `q` is specified, this should be a
            `ScalarMappable` to map `q` values to colors.
        label : str, optional
            Label to add on a plot legend for this trajectory
        info_text : str, optional
            Additional text to display at the bottom of the figure.
        horizon : float or numpy.ndarray, optional
            If provided, plots a horizon as a circle with this radius.
        horizon_param : dict, optional
            Additional parameters for the horizon patch (e.g., `color`, `alpha`).
        projection : str or numpy.ndarray, default: 'xy'
            Projection plane or custom projection matrix.
        ax : matplotlib.axes.Axes, optional
            A matplotlib axes object to plot on. If `None`, a new one is created.
        fig : matplotlib.figure.Figure, optional
            A matplotlib figure object. If `None`, a new one is created.
        **kwargs
            Additional keyword arguments are passed to the underlying plotting
            function (`_plot_traj`).
        """
        pass



TRAJ_2D_BACKEND_REGISTRY = CallableRegistry(Traj2DBackend)


def _normalize_trajectories(coords: np.ndarray | List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalizes various trajectory formats into a list of arrays.
    Each array in the list represents a single trajectory.
    """
    if isinstance(coords, list):
        # Assuming it's a list of arrays for ragged trajectories
        return coords
    
    if not isinstance(coords, np.ndarray):
        raise TypeError(f"Unsupported type for coords: {type(coords)}")

    if coords.ndim == 2:  # Single trajectory (N, D)
        return [coords]
    
    if coords.ndim == 3:  # Batched trajectory (B, N, D)
        return [coords[i] for i in range(coords.shape[0])]
        
    if coords.ndim == 1 and coords.dtype == object: # Ragged np.array of objects
        return list(coords)

    raise ValueError(f"Unsupported numpy array shape or type for coords: ndim={coords.ndim}, dtype={coords.dtype}")


def _set_borders(ax: plt.Axes, borders: Optional[Tuple[float, float, float, float]] = None, coords: Optional[List[np.ndarray]] = None, pad: float = 0.05):
    """Helper to set axis borders, either explicitly or from data."""
    if borders is not None:
        ax.set_xlim(borders[0], borders[1])
        ax.set_ylim(borders[2], borders[3])
        return
    if coords is not None and len(coords) > 0:
        all_coords = np.vstack(coords).reshape(-1, 2)
        if all_coords.size == 0:
            return
        x_min, y_min = np.min(all_coords, axis=0)
        x_max, y_max = np.max(all_coords, axis=0)

        x_pad = (x_max - x_min) * pad
        y_pad = (y_max - y_min) * pad
        
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

@TRAJ_2D_BACKEND_REGISTRY.register('mpl', aliases=['matplotlib'])
def _mpl_traj_2d_plotter(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray = None,
    sm: ScalarMappable = None,
    projection: str | np.ndarray = "xy",
    ax: plt.Axes = None,
    fig: plt.Figure = None,
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
    sm : matplotlib.cm.ScalarMappable, optional
        A pre-configured scalar mappable object that defines the color mapping.
    projection : str or numpy.ndarray, default: 'xy'
        The projection to apply to the coordinates.
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

    if q is not None:
        if sm is None:
            raise ValueError("ScalarMappable 'sm' must be provided when 'q' is specified.")
        raise RuntimeError('This method requires refactor')
        # coords_list = _normalize_trajectories(coords_xy)
        # _multicolored_lines_2d(
        #     x=coords_list, c=q, ax=ax, sm=sm, **kwargs,
        # )

    else:
        label = kwargs.pop("label", None)
        color = kwargs.pop("color", None)
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            **kwargs,
        )

    return fig, ax


def plot2d(
    coords: np.ndarray,
    q: np.ndarray = None,
    borders: Optional[Tuple[float, float, float, float]] = None,
    horizon: float | np.ndarray = None,
    horizon_param: Dict[str, Any] = None,
    projection: str | np.ndarray = "xy",
    backend: str = "mpl",
    cmap: str | Colormap = None,
    sm: ScalarMappable = None,
    q_label: str = None,
    label: str = None,
    info_text: str = None,
    ax: plt.Axes = None,
    fig: plt.Figure = None,
    # **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D projection of trajectory data

    Parameters
    ----------
    coords : torch.Tensor
        Trajectory coordinates of shape `(..., steps, 3)` or `(..., steps, 2)`.
    q : torch.Tensor, optional
        Values for coloring the trajectories, of shape `(..., steps)` or
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
    projection : str or numpy.ndarray, default: 'xy'
        Projection plane or custom projection matrix.
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
        if cmap is None:
            raise ValueError("A colormap 'cmap' must be provided to color the trajectory.")
        sm = _get_scalar_mappable([q], cmap=cmap)


    projector = Projector(projection)
    coords_xy = projector.project(coords)
    _set_borders(ax, borders=borders, coords=[coords_xy])

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

    plotter = TRAJ_2D_BACKEND_REGISTRY.get(backend)

    outp = plotter(
        x=coords_xy[..., 0],
        y=coords_xy[..., 1],
        q=q,
        # horizon=horizon,
        # horizon_param=horizon_param

    )

    return outp