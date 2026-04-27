import math

from typing import Dict, Tuple, Iterable, Literal, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from bhtrace.utils.routines import interpolate_curve
from bhtrace.graphics.presets import axis_map

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

    return fig, ax


def add_info_text(fig: plt.Figure, info_text: str):
    """
    Adds informational text to the bottom of a figure.
    """
    if info_text:
        fig.text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize='small')


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
    def __init__(self, projection: Literal["xy", "yz", "xz"] | np.ndarray = "xy"):
        if isinstance(projection, str):
            if len(projection) != 2 or any(ax not in axis_map for ax in projection):
                raise ValueError(
                    f"Projection string must be two characters from 'x', 'y', 'z'., got {projection}"
                )
            self.proj = [axis_map[ax] for ax in projection]
            self.use_matrix = False
        elif isinstance(projection, np.ndarray):
            assert projection.ndim == 2, f"Projection matrix must be 2D, got shape {projection.shape}"
            assert projection.shape[1] == 2, f"Projection matrix must have 2 columns, got shape {projection.shape}"
            self.proj= projection
            self.use_matrix = True
        else:
            raise ValueError("Projection must be a string or a numpy array.")

    def project(self, coords: np.ndarray) -> np.ndarray:
        """Projects a single coordinate array.

        Parameters
        ----------
        coords : np.ndarray
            Input coordinates of shape [..., 3] or [..., 2].

        Returns
        -------
        np.ndarray
            Projected coordinates of shape [..., 2].
        """
        if self.use_matrix:
            return coords @ self.proj
        else:
            return coords[..., self.proj]
