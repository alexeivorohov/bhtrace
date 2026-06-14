"""
This submodule defines some methods and presets common in colored plots, mostly matplotlib-specific.

"""

from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from matplotlib.colors import Colormap, Normalize
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable


def _single_rgba_vs_coords(rgba: np.ndarray, coords: np.ndarray, i: int = None) -> np.ndarray:
    """
    
    Parameters
    ----------
    rgba : np.ndarray of shape (N, C) or (1, C)
        Input rgba colors.
    xy : np.ndarray of shape (N, D)
    
    Returns
    -------
    np.ndarray
    """ 
    if rgba.shape[0] == 1:
        return rgba.repeat(coords.shape[0], 0)
    if rgba.shape[0] == coords.shape[0]:
        return rgba
    raise ValueError(
        f"Shape mismatch: {rgba.shape} non broadcastable to {coords.shape}"
    )


def _normalize_rgba_to_list(rgba: np.ndarray | List[np.ndarray], coords_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Produces list of color arrays, consistent to the list of array coordinates

    Parameters
    ----------
    rgba : np.ndarray or list of np.ndarray
        Input rgba colors. Expected as list of np.ndarrays or 

    Return
    ------
    list of np.ndarray

    """
    n = len(coords_list)
    if isinstance(rgba, np.ndarray):
        if rgba.ndim == 3: # batched colors for each step
            rgba = [rgba[i, ...] for i in range(n)]
        elif rgba.ndim == 2: # one color for each trajectory
            rgba = [rgba[i, np.newaxis, ...] for i in range(n)]
        elif rgba.ndim == 1: # one color for all trajectories
            rgba = [rgba[np.newaxis, ...]] * len(coords_list)
        else:
            raise ValueError(
                f"Usupported dimensity of `rgba` ({rgba.ndim})"
            )

    if isinstance(rgba, list) and len(rgba) == len(coords_list):            
        idx = list(range(len(coords_list)))
        return list(map(_single_rgba_vs_coords, rgba, coords_list, idx))
        
    raise ValueError(
        f"Unsupported type or len of `rgba`: type={type(rgba)}, len={len(rgba)}"
    )


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


def _multicolored_line_2d_single(
    xy: np.ndarray, 
    rgba: np.ndarray, 
    ax: plt.Axes,
    linewidth: int,
    **kwargs
) -> None:
    """
    Plot a line with a color specified along the line by a third value.

    Mostly follows matplotlib example:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    if len(xy) > 1:
        pass

    xy_midpts = 0.5*(xy[1:, :] + xy[:-1, :])
    coord_start = xy[:-1, np.newaxis, :]
    coord_mid = xy_midpts[:, np.newaxis, :]
    coord_end = xy[1:, np.newaxis, :]

    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, linewidth=linewidth, **kwargs)
    # c has shape (N,), segments has shape (N-1, ...).
    # We color segments by the color of the starting point.
    lc.set_color(rgba[:-1])
    ax.add_collection(lc)


def _multicolored_lines_2d(
    xy_list: List[np.ndarray],
    rgba_list: List[np.ndarray],
    ax: plt.Axes,
    linewidth: int = 2,
    **lc_kwargs,
) -> None:
    """Plots multiple colored 2d lines

    Parameters
    ----------
    xy_list : List[np.ndarray]
        Line coordinates. Expected to be a list of 2D numpy arrays,
        each of shape (N_i, 2).
    rgba_list : List[np.ndarray]
        Array of rgba color values. Can be:
        - List of 1D numpy arrays, each of shape (N_i, 4) for per-point coloring.
        - Single 1D numpy array of shape (B, 4) for per-line coloring (broadcast).
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    """
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    for xy_, rgba_ in zip(xy_list, rgba_list):
        _multicolored_line_2d_single(
            xy=xy_, rgba=rgba_, ax=ax, linewidth=linewidth, **default_kwargs,
        )


def _multicolored_line_3d_single(
    xyz: np.ndarray, 
    rgba: np.ndarray, 
    ax: plt.Axes,
    linewidth: int,
    **kwargs
) -> None:
    """
    """
    xyz_midpts = 0.5*(xyz[1:, :] + xyz[:-1, :])
    coord_start = xyz[:-1, np.newaxis, :]
    coord_mid = xyz_midpts[:, np.newaxis, :]
    coord_end = xyz[1:, np.newaxis, :]

    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    lc = Line3DCollection(segments, linewidth=linewidth, **kwargs)
    lc.set_color(rgba[:-1])
    ax.add_collection(lc)


def _multicolored_lines_3d(
    xyz_list: List[np.ndarray],
    rgba_list: List[np.ndarray],
    ax: plt.Axes,
    linewidth: int = 2,
    **lc_kwargs,
) -> None:
    """Plots multiple colored 3d lines

    Parameters
    ----------
    xyz_list : List[np.ndarray]
        Line coordinates. Expected to be a list of 2D numpy arrays,
        each of shape (N_i, 3).
    rgba_list : List[np.ndarray]
        Array of rgba color values. Can be:
        - List of 1D numpy arrays, each of shape (N_i, 4) for per-point coloring.
        - Single 1D numpy array of shape (B, 4) for per-line coloring (broadcast).
    ax : Axes
        Axes object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.
    """
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    for xyz_, rgba_ in zip(xyz_list, rgba_list):
        _multicolored_line_3d_single(
            xyz=xyz_, rgba=rgba_, ax=ax, linewidth=linewidth, **default_kwargs,
        )
