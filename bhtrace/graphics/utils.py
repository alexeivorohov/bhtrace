from collections import defaultdict
import math

from typing import Dict, Tuple, Iterable, Literal, TYPE_CHECKING, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from bhtrace.utils.routines import interpolate_curve
from bhtrace.graphics.presets import axis_map
from mpl_toolkits.mplot3d import Axes3D

DEFAULT_FIGSIZE: Tuple[int, int] = (10, 10)
"""Default figure size for 3D plots."""


def opt_mosaic(ax_ids: Iterable, fill_None=True, filler=None, rot=False):
    '''
    Function that composes graphs from list to a visually optimal mosaic

    Returns: 
    - shape: tuple(h_n, w_n)
    - mosaic: nested list 
    '''
    shape_cases = {
        1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (3, 2),
        6: (3, 2), 7: (4, 2), 8: (4, 2), 9: (3, 3)
    }
    ax_ids_list = list(ax_ids)
    n_graphs = len(ax_ids_list)
    shape = shape_cases.get(n_graphs)
    if shape is None:
        side = int(np.ceil(np.sqrt(n_graphs)))
        shape = (side, side)

    mosaic = []
    for h in range(shape[1]):
        row = []
        for w in range(shape[0]):
            i = w + h * shape[0]
            if i < n_graphs:
                row.append(ax_ids_list[i])
            elif fill_None:
                row.append(filler)
        mosaic.append(row)

    if rot:
        mosaic = list(zip(*mosaic))
        shape = (shape[1], shape[0])

    return shape, mosaic


def figure_handler(
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    projection: str = None,
    **subplots_kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Handles figure and axes creation for plots.

    This method allows for flexible plot creation by handling different scenarios:
    - If no fig or ax are passed, a new figure and axes are created.
    - If a fig is provided, a new axes is added to this figure.
    - If an ax is provided, it is used for plotting.

    Returns:
        A tuple containing the figure and axes for the plot.
    """
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(subplot_kw={'projection': projection} if projection else None, **subplots_kwargs)
        else:
            ax = fig.add_subplot(projection=projection)
    else:
        fig = ax.get_figure()
        if projection == '3d':
            ax = Axes3D(ax)

    return fig, ax


def add_info_text(fig: plt.Figure, info_text: str):
    """
    Adds informational text to the bottom of a figure.
    """
    if info_text:
        fig.text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize='small')

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

def _value_cleaning_(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v[mask.__neg__()] = np.nan

    return v

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

projection_map = {
    "t": {4: 0},
    "x": {2: 0, 3: 0, 4: 1},
    "y": {2: 1, 3: 1, 4: 2},
    "z": {3: 2, 4: 3},
    "0": defaultdict(lambda : 0),
    "1": defaultdict(lambda : 1),
    "2": defaultdict(lambda : 2),
    "3": defaultdict(lambda : 3),
}
class Projector:
    """Creates a reusable projector for coordinate transformations.

    This class pre-processes a projection definition (e.g., 'xy' or a matrix)
    to create an efficient projection operator that can be applied to one or
    more coordinate arrays.

    Parameters
    ----------
    projection : {'xy', 'yz', 'xz'} or numpy.ndarray, default: 'xy'
        The plane to project onto. Can be one of 'xy', 'yz', 'xz', or a
        custom 3x2 projection matrix.
    """
    def __init__(self, projection: str | np.ndarray = "xy"):
        if isinstance(projection, str):
            if len(projection) != 2 or any(ax not in projection_map for ax in projection):
                raise ValueError(
                    f"Projection string must be two characters from  {projection_map.keys()}, got {projection}"
                )
            self.proj = {i: [projection_map[ax][i] for ax in projection] for i in range(2, 5)}
            self.use_matrix = False
        elif isinstance(projection, np.ndarray):
            assert projection.ndim == 2, f"Projection matrix must be 2D, got shape {projection.shape}"
            assert projection.shape[1] == 2, f"Projection matrix must have 2 columns, got shape {projection.shape}"
            self.proj = projection
            self.use_matrix = True
        else:
            raise ValueError("Projection must be a string or a numpy array.")

    def project(self, coords: np.ndarray) -> np.ndarray:
        """Projects a single coordinate array.

        Parameters
        ----------
        coords : np.ndarray
            Input coordinates of shape [..., 4], [..., 3] or [..., 2].

        Returns
        -------
        Tuple[np.ndarray]
            Projected coordinates
        """
        if self.use_matrix:
            return coords @ self.proj
        else:
            proj = self.proj[coords.shape[-1]]
            # raise ValueError(f'Projection is {proj}')
            return coords[..., proj]

if __name__ == '__main__':
    
    proj = Projector('xy')
    print(proj.proj)