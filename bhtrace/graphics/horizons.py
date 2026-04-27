import math

from typing import Dict, Tuple, Iterable, Literal, TYPE_CHECKING

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from bhtrace.utils.routines import interpolate_curve

def horizon_patch_2d(
        horizon: float | np.ndarray, 
        h_x: float = 0, 
        h_y: float = 0,
        interpolation_params: Dict = None,
    ) -> patches.Patch:

    if isinstance(horizon, (float, int)):
        patch = patches.Circle((h_x, h_y), horizon, edgecolor='black', facecolor='black', lw=2)
    else:
        if interpolation_params is not None:
            horizon = interpolate_curve(
                x=horizon[..., 0], y=horizon[..., 1], **interpolation_params
            )
            horizon = np.stack(horizon, axis=-1)
        horizon[..., 0] += h_x
        horizon[..., 1] += h_y
        patch = patches.Polygon(horizon, closed=True, edgecolor='black', facecolor='black', lw=2)
    
    return patch


def horizon_surface_3d(
        horizon: float | np.ndarray, 
        h_x: float = 0, 
        h_y: float = 0,
        h_z: float = 0,
        interpolation_params: Dict = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(horizon, (float, int)):
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20))
        x = h_x + horizon * np.cos(u) * np.sin(v)
        y = h_y + horizon * np.sin(u) * np.sin(v)
        z = h_z + horizon * np.cos(v)
    else:
        if interpolation_params is not None:
            horizon = interpolate_curve(
                x=horizon[..., 0], y=horizon[..., 1], **interpolation_params
            )
            horizon = np.stack(horizon, axis=-1)
        x = h_x + horizon[..., 0]
        y = h_y + horizon[..., 1]
        z = h_z + horizon[..., 2]
    
    return x, y, z