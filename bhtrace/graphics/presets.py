
from typing import Dict, Tuple, Iterable, TYPE_CHECKING

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import math


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

class Coloring:
    """
    A collection of methods for generating color schemes for plots.
    """
    @classmethod
    def lensing_colors(cls, n_windings: torch.Tensor):
        """
        Assigns colors based on the number of windings 'n'.
        - n < 0.75: direct (blue)
        - 0.75 <= n < 1.25: lensed (orange)
        - n >= 1.25: ring (red)
        """
        n_windings_np = n_windings.cpu().numpy().flatten()
        conditions = [
            n_windings_np < 0.75,
            (n_windings_np >= 0.75) & (n_windings_np < 1.25)
        ]
        choices = ['blue', 'orange']
        colors = np.select(conditions, choices, default='red')
        return colors

    @classmethod
    def trajectory_colors(cls, n_traj: int, cmap: str = 'viridis') -> list:
        """
        Generates a list of colors for plotting multiple trajectories using a colormap.
        """
        cmap_func = plt.get_cmap(cmap)
        return [cmap_func(i) for i in np.linspace(0, 1, n_traj)]

    @classmethod
    def image_colors(cls):
        """
        Placeholder for image coloring schemes.
        """
        pass



