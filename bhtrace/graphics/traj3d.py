"""
3D plotting utilities.
"""

from typing import Dict, Tuple, Iterable, List, Optional, Any, Literal, Protocol

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from matplotlib.cm import ScalarMappable


from bhtrace.graphics.utils import figure_handler, _dir_handler, add_info_text, _normalize_trajectories
from bhtrace.graphics.coloring import _multicolored_lines_3d, _get_scalar_mappable, _normalize_rgba_to_list
from bhtrace.utils import CallableRegistry

class Plot3DBackend(Protocol):

    def __call__(
        xyz: np.ndarray,
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
        xyz : np.ndarray | list of np.ndarray
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


PLOT3D_BACKEND_REGISTRY = CallableRegistry(Plot3DBackend)

@PLOT3D_BACKEND_REGISTRY.register('mpl', aliases=['matplotlib'])
def _mpl_traj_3d_plotter(
    xyz: np.ndarray,
    rgba: Optional[np.ndarray],
    ax: Optional[plt.Axes],
    fig: Optional[plt.Figure],
    label: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot particle trajectories in 3D, using matplotlib

    This is a low-level plotting function.

    Parameters
    ----------
    xyz : numpy.ndarray or list of numpy.ndarray
        Trajectory coordinates. Can be a single trajectory (N, 3), a batch (B, N, 3),
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
        
        xyz_list = _normalize_trajectories(xyz)
        rgba_list = _normalize_rgba_to_list(rgba, xyz_list)
        _multicolored_lines_3d(
            xyz_list=xyz_list, rgba_list=rgba_list, ax=ax, **kwargs,
        )

    else:
        ax.plot(
            xyz[..., 0].T,
            xyz[..., 1].T,
            xyz[..., 2].T,
            label=label,
            # color=color,
            **kwargs,
        )

    return fig, ax


def plot3d(
    xyz: np.ndarray,
    q: np.ndarray = None,
    borders: float = 20,
    horizon: float | np.ndarray = None,
    horizon_param: Dict[str, Any] = None,
    backend: str = "mpl",
    cmap: str | Colormap = None,
    sm: ScalarMappable = None,
    q_label: str = None,
    q_scale: str = 'linear',
    label: str = None,
    info_text: str = None,
    elev: float = 45.0,
    azim: float = -45.0,
    roll: float = 0.0,
    ax: plt.Axes = None,
    fig: plt.Figure = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 2D projection of trajectory data

    Parameters
    ----------
    xyz : np.ndarray
        Trajectory coordinates of shape `(..., steps, 3)`
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
    elev : float, default=45
        The elevation angle in degrees.
    azim : float, default=45
        The azimuth angle in degrees.
    roll : float, default=0
        The roll angle in degrees.
    **kwargs
        Additional keyword arguments passed to `ax.plot` or `Line3DCollection`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    """
    fig, ax = figure_handler(fig, None, projection='3d')

    if q is not None and sm is None:
        if cmap is None:
            raise ValueError("A colormap 'cmap' must be provided to color the trajectory.")
        sm = _get_scalar_mappable([q], cmap=cmap)
        rgba = sm.to_rgba(q.squeeze())
    else:
        rgba = None

    # _set_borders(ax, borders=borders, coords=xyz)
    if horizon is not None:
        pass
        # TODO: Handle horizons correctly (i guess it will require a separate class)
        # horizon_param = horizon_param or {}
        # horizon_patch = horizon_patch_2d(horizon, **horizon_param)
        # ax.add_patch(horizon_patch)
    
    if sm is not None:
        fig.colorbar(sm, ax=ax, label=q_label)

    if info_text:
        add_info_text(fig, info_text)
 

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-borders, borders)
    ax.set_ylim(-borders, borders)
    ax.set_zlim(-borders, borders)

    ax.set_aspect("equal", "box")
    ax.grid(True)

    if label:
        ax.legend()

    outp = PLOT3D_BACKEND_REGISTRY[backend](
        xyz=xyz,
        rgba=rgba,
        ax=ax,
        fig=fig,
        **kwargs,
    )

    return outp

# def _scatter3d_backend_mpl()
def _align_xyz_and_q_scatter(
    xyz: np.ndarray, 
    q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align and flatten xyz and q arrays for scatter plotting.

    Parameters
    ----------
    xyz : np.ndarray
        Array of cartesian coordinates of shape (..., 3)
    q : np.ndarray
        Array of values for coloring, shape (...) or (..., 1)
        If batch dimension (...) is different from the batch dimension of `xyz`,
        attempt to broadcast these arrays will be done.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Flattened arrays with last dimension unchanged.
    """

    if q is None:
        return xyz.reshape(-1, 3), None

    xyz_shape = xyz.shape
    q_shape = q.shape
    try:
        broadcasted_xyz = np.broadcast_to(xyz, (xyz_shape[0],) + q_shape)
        broadcasted_q = np.broadcast_to(q, xyz_shape[:-1] + (1,))
    except ValueError:
        raise ValueError(
            f"Cannot broadcast xyz and q with shapes {xyz_shape} and {q_shape}." 
        )

    flattened_xyz = broadcasted_xyz.reshape(-1, 3)
    flattened_q = broadcasted_q.reshape(-1)

    return flattened_xyz, flattened_q

def scatter3d(
    xyz: np.ndarray,
    q: np.ndarray = None,
    borders: float = 20,
    horizon: float | np.ndarray = None,
    horizon_param: Dict[str, Any] = None,
    backend: str = "mpl",
    cmap: str | Colormap = None,
    sm: ScalarMappable = None,
    q_label: str = None,
    q_scale: str = 'linear',
    label: str = None,
    info_text: str = None,
    elev: float = 45.0,
    azim: float = -45.0,
    roll: float = 0.0,
    ax: plt.Axes = None,
    fig: plt.Figure = None,
    **kwargs,
):
    fig, ax = figure_handler(fig, ax, projection='3d')

    # xyz, q = _align_xyz_and_q_scatter(xyz, q)

    if q is not None and sm is None:
        if cmap is None:
            raise ValueError("A colormap 'cmap' must be provided to color the trajectory.")
        sm = _get_scalar_mappable([q], cmap=cmap)
        rgba = sm.to_rgba(q)
    else:
        rgba = None

    # _set_borders(ax, borders=borders, coords=xyz)
    if horizon is not None:
        pass
        # TODO: Handle horizons correctly (i guess it will require a separate class)
        # horizon_param = horizon_param or {}
        # horizon_patch = horizon_patch_2d(horizon, **horizon_param)
        # ax.add_patch(horizon_patch)
    
    if sm is not None:
        fig.colorbar(sm, ax=ax, label=q_label)


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-borders, borders)
    ax.set_ylim(-borders, borders)
    ax.set_zlim(-borders, borders)

    ax.set_aspect("equal", "box")
    ax.grid(True)

    if label:
        ax.legend()

    ax.scatter(
        xyz[..., 0],
        xyz[..., 1],
        xyz[..., 2],
        c=rgba,
        **kwargs,
    )

    return fig, ax


# def vector_field(
#     points: torch.Tensor,
#     vectors: torch.Tensor,
#     values: Optional[torch.Tensor] = None,
#     elev: Optional[float] = None,
#     azim: Optional[float] = None,
#     roll: Optional[float] = None,
#     fig: Optional[plt.Figure] = None,
#     ax: Optional[plt.Axes] = None,
#     **kwargs,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """Plot a 3D vector field.

#     Parameters
#     ----------
#     points : torch.Tensor
#         Origins of the vectors, shape `(..., 3)`.
#     vectors : torch.Tensor
#         Vector components, shape `(..., 3)`.
#     values : torch.Tensor, optional
#         Scalar values for coloring the vectors.
#     elev : float, optional
#         Elevation angle of the camera.
#     azim : float, optional
#         Azimuth angle of the camera.
#     roll : float, optional
#         Roll angle of the camera.
#     fig : matplotlib.figure.Figure, optional
#         Figure to plot on. If `None`, a new one is created.
#     ax : matplotlib.axes.Axes, optional
#         Axes to plot on. If `None`, a new one is created.
#     **kwargs
#         Additional keyword arguments passed to `ax.quiver`.

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         The figure object for the plot.
#     ax : matplotlib.axes.Axes
#         The axes object for the plot.
#     """
#     fig, ax = figure_handler(fig, ax)
#     _dir_handler(ax, elev=elev, azim=azim, roll=roll)

#     x = points[..., 0].view(-1)
#     y = points[..., 1].view(-1)
#     z = points[..., 2].view(-1)

#     u = vectors[..., 0].view(-1)
#     v = vectors[..., 1].view(-1)
#     w = vectors[..., 2].view(-1)

#     # Pop cmap from kwargs if we are coloring by values
#     cmap_name = kwargs.pop("cmap", None) if values is not None else None

#     q = ax.quiver(x, y, z, u, v, w, **kwargs)

#     if values is not None:
#         norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
#         cmap = plt.get_cmap(cmap_name or "viridis")
#         colors = cmap(norm(values.cpu().numpy()))
#         q.set_facecolors(colors.reshape(-1, 4))

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

#     return fig, ax
