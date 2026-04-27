from typing import List, Dict, Optional, Tuple, Literal, Protocol
\
import torch
import numpy as np
from scipy.signal import find_peaks


import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import bhtrace.graphics.uniplot_wraps as uniplot
from bhtrace.graphics.utils import add_info_text, figure_handler
from bhtrace.utils.registry import CallableRegistry


__curve_style__ = {
    'alpha': 0.9,
    'zorder': 1,
}

__curve_p_style__ = {
    'alpha': 0.5
}

__peak_style__ = {
    'linestyle': 'dashed',
    # 'marker': 'x',
    'zorder': 3,
    'alpha': 0.4
}

__peak_p_style__ = {
    'marker': 'x',
    'zorder': 1,
    'alpha': 0.5
}
class LensingBackend(Protocol):
    def __call__(
        dphi: np.ndarray | Dict[str, np.ndarray],
        b: np.ndarray | Dict[str, np.ndarray],
        e_b: np.ndarray | Dict[str, np.ndarray ],
        show_peaks: bool,
        show_points: bool,
        add_2dplot: bool,
        peak_dist: int,
        ax: plt.Axes,
        fig: plt.Figure,
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        ...

LENSING_BACKENDS_REGISTRY = CallableRegistry(LensingBackend)

def lensing_curve(
    b: np.ndarray | Dict[str, np.ndarray],
    dphi: np.ndarray | Dict[str, np.ndarray],
    show_peaks: bool = True,
    show_points: bool = False,
    windings: bool = False,
    label: str = None,
    color: str = None,
    sm: 'ScalarMappable' = None,
    minimal_peak_distance: int = 2,
    backend: str = 'matplotlib',
    ax: plt.Axes = None,
    fig: plt.Figure = None,
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Preset for plotting declination angle vs. impact parameter curve.

    Parameters
    ----------
    dphi: np.ndarray | Dict[np.ndarray] - inclination angles. 
        If dict passed, lensing curves will be displayed on the same axes and labeled with keys of this dict.
    b: np.ndarray | Dict[np.ndarray] - impact factors
        For dict behaviour is same as for dphi.
    e_b: np.ndarray | Dict[np.ndarray] - unit vector in the direction of impact factor increase.
    show_peaks: bool - determine and note peaks of lensing function (default: true)
    show_points: bool - show points of the curve (default: false)
    peak_dist: int - minimum distance between peaks, passed to scipy find_peaks (default: 2)
    ax: plt.Axes - axes to plot on
    fig: plt.Figure - figure to plot on
    
    """

    if windings:
        dphi = dphi / np.pi / 2

    if show_peaks:
        peak_idxs, _ = find_peaks(dphi, distance=minimal_peak_distance)
    else:
        peak_idxs = None


    plotter = LENSING_BACKENDS_REGISTRY[backend]

    return plotter(
        dphi = dphi,
        b = b,
        peak_idxs = peak_idxs,
        label = label,
        sm = sm,
        color = color,
        show_points = show_points,
        ax = ax,
        fig = fig,
    )

@LENSING_BACKENDS_REGISTRY.register('matplotlib', aliases=['mpl'])
def _lensing_backend_mpl(
    b: np.ndarray,
    dphi: np.ndarray,
    peak_idxs: Optional[np.ndarray] = None,
    color: str = None,
    sm: ScalarMappable = None,
    show_points: bool = False,
    label: str = None,
    ax: plt.Axes = None,
    fig: plt.Figure = None,
) -> Tuple[plt.Figure, plt.Axes]:
    print('Plotter called')

    fig, ax = figure_handler(fig, ax)

    if peak_idxs is not None:
        for peak in peak_idxs.tolist():
            ax.scatter(b[peak], dphi[peak], color=color, **__peak_p_style__)
            ax.axvline(b[peak], color=color, **__peak_style__)

    if sm is not None:
        c = sm.to_rgba(dphi)
        ax.plot(b, dphi, color=color, label=label, **__curve_style__)
    else:
        ax.plot(b, dphi, color=color, label=label, **__curve_style__)

    if show_points:
        ax.scatter(b, dphi, color=color, label=None, **__curve_p_style__)
        
    return fig, ax

if __name__ == '__main__':

    b = np.linspace(0, 10, 64)
    b_c = 3*np.sqrt(3)
    dphi = 2*np.pi/np.sqrt(np.abs(1 - b/b_c)+0.1)

    fig, ax = lensing_curve(b, dphi)
    
    plt.show()