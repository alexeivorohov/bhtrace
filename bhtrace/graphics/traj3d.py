"""
3D plotting utilities.
"""

from typing import Tuple, List, Optional, Any, Dict, Iterable

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection

DEFAULT_FIGSIZE: Tuple[int, int] = (10, 10)
"""Default figure size for 3D plots."""


def _fig_handler(
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Handle figure and axes creation for 3D plots.

    If no `ax` is provided, a new 3D subplot is created on a new or existing `fig`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        The figure to which the plot will be added.
    ax : matplotlib.axes.Axes, optional
        The axes on which the plot will be drawn.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The 3D axes object for the plot.
    """
    if ax is None:

        if fig is None:
            fig = plt.figure(figsize=DEFAULT_FIGSIZE)
        ax = fig.add_subplot(projection="3d")
    else:
        # set axes to 3D if not already
        # if not isinstance(ax, mpl.axes._subplots.Axes3DSubplot):
        #     raise ValueError("Provided ax must be a 3D subplot.")
        fig = ax.get_figure()
    return fig, ax


def _dir_handler(
    ax: plt.Axes,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    roll: Optional[float] = None,
):
    """Handle the viewing direction of the 3D plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to apply the view direction to.
    elev : float, optional
        The elevation angle in degrees.
    azim : float, optional
        The azimuth angle in degrees.
    roll : float, optional
        The roll angle in degrees.
    """
    ax.view_init(elev=elev, azim=azim, roll=roll)


def plot3d(
    coords: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    q_label: Optional[str] = None,
    color: Optional[Any] = None,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    roll: Optional[float] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 3D trajectory.

    This function can plot a single trajectory or a batch of trajectories. It
    supports gradient coloring along the line if a quantity `q` is provided.

    Parameters
    ----------
    coords : torch.Tensor
        Trajectory coordinates of shape `(steps, 3)` or `(batch, steps, 3)`.
    q : torch.Tensor, optional
        Values for coloring the trajectory, of shape `(steps,)`. If provided,
        `color` should be a colormap.
    q_label : str, optional
        Label for the colorbar if `q` is provided.
    color : any, optional
        Color specification. Can be a matplotlib color string, or a colormap
        if `q` is provided.
    label : str, optional
        Label for the trajectory, to be shown in a legend.
    ax : matplotlib.axes.Axes, optional
        An axes object to plot on. If `None`, a new one is created.
    fig : matplotlib.figure.Figure, optional
        A figure object. If `None`, a new one is created.
    elev : float, optional
        The elevation angle in degrees.
    azim : float, optional
        The azimuth angle in degrees.
    roll : float, optional
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
    fig, ax = _fig_handler(fig, ax)
    _dir_handler(ax, elev=elev, azim=azim, roll=roll)

    coords_np = coords.cpu().numpy()
    
    if coords_np.ndim == 2:
        coords_np = np.expand_dims(coords_np, 0)

    for i, c_np in enumerate(coords_np):
        # Use trajectory index for label if multiple are plotted
        line_label = f"{label} {i}" if label and len(coords_np) > 1 else label

        if q is not None:
            points = c_np.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            q_np = q.cpu().numpy().flatten()
            c_val = (q_np[:-1] + q_np[1:]) / 2

            norm = plt.Normalize(c_val.min(), c_val.max())
            lc = Line3DCollection(segments, cmap=color, norm=norm, **kwargs)
            lc.set_array(c_val)
            ax.add_collection(lc)

            if line_label:
                # Add proxy artist for legend
                ax.plot([], [], [], color=color(norm(np.mean(c_val))), label=line_label)

            if q_label and i == 0:  # Add colorbar only once
                mappable = plt.cm.ScalarMappable(cmap=color, norm=norm)
                fig.colorbar(mappable, ax=ax, label=q_label, shrink=0.6)
        else:
            ax.plot(
                c_np[:, 0],
                c_np[:, 1],
                c_np[:, 2],
                color=color,
                label=line_label,
                **kwargs,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if label:
        ax.legend()

    return fig, ax


def point_cloud(
    points: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    roll: Optional[float] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 3D point cloud.

    Parameters
    ----------
    points : torch.Tensor
        Points to draw of shape `(..., 3)`.
    values : torch.Tensor, optional
        Scalar values for each point, used for coloring.
    elev : float, optional
        Elevation angle of the camera.
    azim : float, optional
        Azimuth angle of the camera.
    roll : float, optional
        Roll angle of the camera.
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If `None`, a new one is created.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If `None`, a new one is created.
    **kwargs
        Additional keyword arguments passed to `ax.scatter`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    """
    fig, ax = _fig_handler(fig, ax)
    _dir_handler(ax, elev=elev, azim=azim, roll=roll)

    x = points[..., 0].view(-1)
    y = points[..., 1].view(-1)
    z = points[..., 2].view(-1)

    c = values.view(-1) if values is not None else None

    ax.scatter(x, y, z, c=c, marker=".", **kwargs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax


def vector_field(
    points: torch.Tensor,
    vectors: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    roll: Optional[float] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 3D vector field.

    Parameters
    ----------
    points : torch.Tensor
        Origins of the vectors, shape `(..., 3)`.
    vectors : torch.Tensor
        Vector components, shape `(..., 3)`.
    values : torch.Tensor, optional
        Scalar values for coloring the vectors.
    elev : float, optional
        Elevation angle of the camera.
    azim : float, optional
        Azimuth angle of the camera.
    roll : float, optional
        Roll angle of the camera.
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If `None`, a new one is created.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If `None`, a new one is created.
    **kwargs
        Additional keyword arguments passed to `ax.quiver`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    """
    fig, ax = _fig_handler(fig, ax)
    _dir_handler(ax, elev=elev, azim=azim, roll=roll)

    x = points[..., 0].view(-1)
    y = points[..., 1].view(-1)
    z = points[..., 2].view(-1)

    u = vectors[..., 0].view(-1)
    v = vectors[..., 1].view(-1)
    w = vectors[..., 2].view(-1)

    # Pop cmap from kwargs if we are coloring by values
    cmap_name = kwargs.pop("cmap", None) if values is not None else None

    q = ax.quiver(x, y, z, u, v, w, **kwargs)

    if values is not None:
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        cmap = plt.get_cmap(cmap_name or "viridis")
        colors = cmap(norm(values.cpu().numpy()))
        q.set_facecolors(colors.reshape(-1, 4))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax
