import warnings
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection, Collection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from matplotlib.colors import Colormap, Normalize
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable


def _multicolored_line_2d_single(
    x: np.ndarray, 
    c: np.ndarray, 
    ax: plt.Axes,
    linewidth: int,
    sm: ScalarMappable,
    **kwargs
) -> Collection:
    """
    Plot a line with a color specified along the line by a third value.

    Mostly follows matplotlib example:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    x_midpts = 0.5*(x[1:, :] + x[:-1, :])
    coord_start = x[:-1, np.newaxis, :]
    coord_mid = x_midpts[:, np.newaxis, :]
    coord_end = x[1:, np.newaxis, :]

    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, linewidth=linewidth, **kwargs)
    # c has shape (N,), segments has shape (N-1, ...).
    # We color segments by the color of the starting point.
    lc.set_color(sm.to_rgba(c[:-1]))
    
    return ax.add_collection(lc)


def _get_scalar_mappable(
    c_list: List[np.ndarray],
    cmap: str | Colormap,
    norm: Normalize = None,
) -> ScalarMappable:
    """
    Creates a ScalarMappable for coloring based on a list of color values.
    If norm is not provided, it is automatically determined from the data.
    """
    cmap = plt.get_cmap(cmap)
    cmap.set_over('m')

    if norm is None:
        c_vals_to_concat = [c_ for c_ in c_list if c_ is not None and len(c_) > 0]
        if c_vals_to_concat:
            all_c_vals = np.concatenate(c_vals_to_concat)
        else:
            all_c_vals = np.array([])
        if all_c_vals.size > 0:
            norm = plt.Normalize(vmin=all_c_vals.min(), vmax=all_c_vals.max())
        else:
            # If all_c_vals is empty, then no data to normalize, use default Normalize
            norm = plt.Normalize()

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    return sm


def _multicolored_lines_2d(
        x: List[np.ndarray], # Changed type hint
        c: List[np.ndarray] | np.ndarray, # Changed type hint
        ax: plt.Axes,
        sm: ScalarMappable,
        linewidth: int = 2,
        **lc_kwargs,
    ) -> ScalarMappable:
    """Plots multiple colored 2d lines

    Parameters
    ----------
    x : List[np.ndarray]
        Line coordinates. Expected to be a list of 2D numpy arrays,
        each of shape (N_i, 2).
    c : List[np.ndarray] | np.ndarray
        Array of values, passed to colormap. Can be:
        - List of 1D numpy arrays, each of shape (N_i,) for per-point coloring.
        - Single 1D numpy array of shape (B,) for per-line coloring (broadcast).

    ax : Axes
        Axis object on which to plot the colored line.
    sm : matplotlib.cm.ScalarMappable
        A pre-configured scalar mappable object that defines the color mapping.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        The generated scalar mappable, which can be used to create a colorbar.

    """
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    x_list = x # x is now guaranteed to be a list
    c_list = []

    # c can be a list of arrays or a single array to be broadcasted
    if isinstance(c, list):
        if len(x_list) != len(c):
            raise ValueError(f"For ragged input lengths of x and c must match, got {len(x_list)} and {len(c)}")
        c_list = c
    elif isinstance(c, np.ndarray) and c.ndim == 1 and len(c) == len(x_list):
        # c has shape (B,), x is a list of B arrays
        # broadcast c to each line
        c_list = [np.full(len(x_i), c_i) for x_i, c_i in zip(x_list, c)]
    elif isinstance(c, np.ndarray) and len(x_list) == 1:
        c_list = [c]
    else:
        raise ValueError("Invalid shape for c with list x input. Expected list of np.ndarray or a 1D np.ndarray for broadcasting.")

    for x_, c_ in zip(x_list, c_list):
        if len(x_) > 1:
            _multicolored_line_2d_single(
                x=x_, c=c_, ax=ax, linewidth=linewidth, sm=sm, **default_kwargs,
            )

    return sm


def _multicolored_line_3d_single(
    x: np.ndarray, 
    c: np.ndarray, 
    ax: plt.Axes,
    linewidth: int,
    sm: ScalarMappable,
    **kwargs
) -> Tuple[plt.Axes, plt.Figure]:
    """
    """

    x_midpts = 0.5*(x[1:, :] + x[:-1, :])
    coord_start = x[:-1, np.newaxis, :]
    coord_mid = x_midpts[:, np.newaxis, :]
    coord_end = x[1:, np.newaxis, :]

    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = Line3DCollection(segments, linewidth=linewidth, **kwargs)
    # c has shape (N,), segments has shape (N-1, ...).
    # We color segments by the color of the starting point.
    lc.set_color(sm.to_rgba(c[:-1]))

    ax.add_collection(lc)
    return ax.get_figure(), ax



