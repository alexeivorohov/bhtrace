"""
This file contains definitions for histogram plots
"""
from typing import List, Optional, Tuple, Literal, Protocol

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import numpy as np
import bhtrace.graphics.uniplot_wraps as uplt

from bhtrace.graphics.utils import add_info_text, figure_handler
from bhtrace.utils.registry import CallableRegistry
from bhtrace.graphics.scaling import scale

# TODO: promote args
# TODO: enhance visuals
# TODO: implement tests
# TODO: review uniplot backends
# TODO: update docs

class HistogramBackend(Protocol):
    def __call__(
        dist: np.ndarray,
        bins: Optional[int | np.ndarray],
        label: Optional[str],
        info_text: Optional[str],
        ax: Optional[plt.Axes],
        fig: Optional[plt.Figure],
        **kwargs,
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Plots 1d histogram
        
        Parameters
        ----------
        dist : np.ndarray
            Probability density or bin count, optionally scaled
        bins : int or np.ndarray
            Histogram bins
        label : str, optional
            Label for the plot.
        info_text : str, optional
            Additional text to display on the plot.
        ax : plt.Axes, optional
            Matplotlib Axes object.
        fig : plt.Figure, optional
            Matplotlib Figure object.
        """
        pass


class RidgeBackend(Protocol):
    def __call__(
        dist: List[np.ndarray],
        parameter: np.ndarray,
        bins: np.ndarray,
        label: str,
        info_text: str,
        ax: plt.Axes,
        fig: plt.Figure,
        density: bool,
        **kwargs,
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Plots a ridge histogram
         
        Ridge histogram is a collection of histograms, evolving with the change of some parameter
        
        Parameters
        ----------
        dist : List[np.ndarray]
            List of 1-d arrays of probability densities/value counts for histograms.
        parameter : List of floats 
            The parameter which marks different items of `data`, e.g. time or step.
        bins : np.ndarray
            1-d array of bin edges of shape [len(dist[0])+1]
        label : optional, str
            label to place on the plot
        info_text : optional, str
            Additional text to display on the plot.
        ax : optional, plt.Axes
            Matplotlib Axes object to plot diagram on.
        fig : optional, plt.Figure
            Matplotlib Figure object to plot diagram on.
        """
        pass


HISTOGRAM_BACKEND_REGISTRY = CallableRegistry(HistogramBackend)
RIDGE_BACKEND_REGISTRY = CallableRegistry(RidgeBackend)


def hist(
    data: np.ndarray,
    bins: Optional[int | np.ndarray] = 10,
    weights: Optional[np.ndarray] = None,
    density: bool = True,
    bin_scale: str = 'linear',
    dist_scale: str = 'linear',
    label: Optional[str] = None,
    info_text: Optional[str] = None,
    backend: Literal["mpl", "uniplot"] = "mpl",
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    **kwargs,
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """Plots a histogram of the given data

    Parameters
    ----------
    data : np.ndarray
        NumPy array. If 1D, a single histogram is plotted.
        If 2D, one histogram per row is plotted.
        If >2D, the array is flattened and a single histogram is plotted
    label : optional, str
        label to place on the plot
    weights: Optional[np.ndarray] = None,
        Label for the data series.
    bin_scale : Literal['linear', 'log'] (default: 'linear')
        A type of scale to apply to the bin axes. 
        Will be also applied to `bins`, if bins are passed 
    dist_scale : Literal['linear', 'log'] (default: 'linear')
        A type of scale to apply to the evaluated counts/densities.
    backend : {'mpl', 'uniplot'}, default 'mpl'
        Plotting backend to use.
    info_text : str, optional
        Additional text to display on the plot.
    fig : plt.Figure, optional
        Existing figure to plot on.
    ax : plt.Axes, optional
        Existing axes to plot on.
    **kwargs
        Additional keyword arguments passed to the backend.

    Returns
    -------
    Tuple[Optional[plt.Figure], Optional[plt.Axes]]
        Figure and Axes object, or (None, None) for uniplot backend.
    """

    p_label = 'p' if density else 'N'
    
    data = scale(data, bin_scale)    
    if isinstance(bins, np.ndarray):
        bins = scale(bins, bin_scale)

    p, bin_edges = np.histogram(data, bins=bins, density=density, weights=weights)
    p = scale(p, dist_scale)


    return HISTOGRAM_BACKEND_REGISTRY[backend](
        data=data,
        bins=bins,
        weights=weights,
        density=density,
        label=label,
        info_text=info_text,
        ax=ax,
        fig=fig,
        **kwargs,
    )

def _process_ridge_data(
    data: List[np.ndarray],
    bins: np.ndarray,
    weights: Optional[List[np.ndarray]],
    density: bool
) -> List[np.ndarray]:
    
    n = len(data)
    assert isinstance(data, list), (
        "To process data for ridge histogram, `data` should be passed as list of numpy arrays."
        f"Got type(`data`): {type(data)}"
    )

    if weights is not None:
        assert len(weights) == n, (
            "Weights are specified, but the number of weight arrays does not correspond number of data arrays:"
            f"{len(weights) != n}"
        )
        for i in range(n):
            assert weights[i].shape == data[i].shape, f"Shape mismatch for weights and data at index {i}"
    else:
        weights = [None] * n

    dist = [
        np.histogram(data[i], bins=bins, density=density, weights=weights[i])[0]
        for i in range(n)
    ]

    return dist
    
def ridge(
    data: List[np.ndarray],
    parameter: List[float] | np.ndarray = None,
    weights: Optional[List[np.ndarray]] = None,
    bins: int | np.ndarray = 16,
    density: bool = True,
    bin_scale: str = 'linear',
    dist_scale: str = 'linear',
    par_scale: str = 'linear',
    label: Optional[str] = None,
    info_text: Optional[str] = None,
    backend: Literal["mpl", "uniplot"] = "mpl",
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    **kwargs,
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """Displays ridgeline histogram plot 
    
    Shows a distribution evolution over parameter (e.g. time) change.

    Parameters
    ----------
    data : List of np.ndarray
        The data to build distributions from.
    parameter : List of floats or np.ndarray (default: None)
        The parameter which marks different items of `data`, e.g. time or step.
        If not provided, states will be marked by their indexes in list.
    bins : int | np.ndarray (default: 16)
        Parameter for controlling bins of the histogram.
        If int is specified, bins will be sampled automatically, using boundary
        values over all items in `data` and accounting for `bin_scale` parameter.
        If np.ndarray is specified, values outside of this array will be ignored.
    weights: optional, List of np.ndarray
        A weightening values for the data. For every element, shapes of `weight` 
        and `data` should match.  
    density : bool (default: True)
        A flag for np.histogram to calculate probability density instead of value counts.
    scale : Literal['linear', 'log'] (default: 'linear')
        A type of scale to apply to the evaluated counts/densities
    bin_scale : Literal['linear', 'log'] (default: 'linear')
        Same as `scale`, but for bin axes. Ignored if `bins` is specified as np.ndarray
    label : Optional label to mark these data samples
    backend : {'mpl', 'uniplot'}, default 'mpl'
        Plotting backend to use.
    """

    if parameter is None:
        parameter = list(range(len(data)))

    dist = _process_ridge_data(data, bins=bins, weights=weights, density=density)


    return RIDGE_BACKEND_REGISTRY[backend](
        dist=dist,
        parameter=parameter,
        bins=bins,
        label=label,
        info_text=info_text,
        ax=ax,
        fig=fig,
        density=density,
        **kwargs,
    )

@HISTOGRAM_BACKEND_REGISTRY.register("mpl", aliases=["matplotlib"])
def _histogram_mpl_1d(
    dist: np.ndarray,
    bin_edges: np.ndarray,
    label: str,
    info_text: Optional[str],
    ax: Optional[plt.Axes],
    fig: Optional[plt.Figure],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a 1D histogram using matplotlib.
    """
    fig, ax = figure_handler(fig, ax)

    ax.grid(kwargs.pop('grid', True))
    ax.stairs(dist, bin_edges, fill=kwargs.pop('fill', True), label=label, **kwargs)

    if label:
        ax.legend()
    if info_text:
        add_info_text(ax, info_text)

    return fig, ax


@RIDGE_BACKEND_REGISTRY.register("mpl", aliases=["matplotlib"])
def _ridge_mpl_3d(
    dist: List[np.ndarray],
    parameter: List[float],
    bins: np.ndarray,
    label: str,
    info_text: str,
    q_scale: str,
    p_scale: str,
    ax: plt.Axes,
    fig: plt.Figure,
    density: bool,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a ridge histogram using matplotlib, in a 3D view.
    """
    fig, ax = figure_handler(fig, ax, projection='3d')

    verts = []
    zs = np.array(parameter)
    max_freq = 0

    for i in range(len(dist)):
        hist = dist[i]
        if hist.size > 0:
            max_freq = max(max_freq, hist.max())

        v = [(bins[0], 0)]
        for j in range(len(hist)):
            v.append((bins[j], hist[j]))
            v.append((bins[j + 1], hist[j]))
        v.append((bins[-1], 0))
        verts.append(v)

    poly = PolyCollection(verts)

    facecolors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(verts))]
    poly.set_facecolors(facecolors)
    poly.set_alpha(kwargs.get('alpha', 0.7))

    ax.add_collection3d(poly, zs=zs, zdir="y")
    
    ax.set_xscale(q_scale)
    if p_scale == 'log':
        ax.set_zscale("log")

    ax.set_xlabel("Value")
    ax.set_ylabel("Parameter")
    ax.set_zlabel("Density" if density else "Frequency")

    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(zs.min(), zs.max())
    if p_scale != 'log':
        ax.set_zlim(0, max_freq * 1.1 if max_freq > 0 else 1)

    if info_text:
        ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, va="top")

    return fig, ax


# @HISTOGRAM_BACKEND_REGISTRY.register("uniplot")
# def _histogram_uniplot_1d(
#     dist: np.ndarray,
#     bin_edges: np.ndarray,
#     label: Optional[str],
#     info_text: Optional[str],
#     ax: Optional[plt.Axes],
#     fig: Optional[plt.Figure],
#     **kwargs,
# ) -> Tuple[None, None]:
#     """
#     Plots a 1D histogram using uniplot.
#     """
#     if label:
#         print(f"Histogram for: {label}")
#     else:
#         print("Histogram")

#     plot_quantity = dist.flatten()
#     if q_scale == "log":
#         if np.any(plot_quantity <= 0):
#             print(
#                 "Warning: non-positive values present in data. Log scale can't be applied to them. They will be ignored."
#             )
#             plot_quantity = plot_quantity[plot_quantity > 0]
#         plot_quantity = np.log10(plot_quantity)
#         kwargs["xlabel"] = kwargs.get("xlabel", "value") + " (log10 scale)"

#     if not isinstance(bin_edges, int):
#         print("Warning: uniplot backend only supports integer number of bins. Ignoring provided bin edges.")
#         bin_edges = 20
        
#     uniplot_kwargs = kwargs.copy()
#     if 'bins' not in uniplot_kwargs:
#         uniplot_kwargs['bins'] = bin_edges

#     uplt.histogram(plot_quantity, **uniplot_kwargs)

#     if info_text:
#         print(info_text)

#     return None, None


# @RIDGE_BACKEND_REGISTRY.register("uniplot")
# def _ridge_uniplot(
#     dist: List[np.ndarray],
#     parameter: List[float],
#     bins: np.ndarray,
#     label: str,
#     info_text: str,
#     q_scale: str,
#     p_scale: str,
#     ax: plt.Axes,
#     fig: plt.Figure,
#     density: bool,
#     **kwargs,
# ) -> Tuple[None, None]:
#     """
#     Plots an evolution 2D histogram using uniplot, with one histogram per column.
#     """
#     print("Warning: uniplot backend for ridge plots is not fully supported and may produce misleading results.")
    
#     for i, p in enumerate(parameter):
#         hist = dist[i]
        
#         # We need to "un-histogram" the data for uniplot, which is not ideal.
#         # This is a crude approximation. Scale to make sure values are not truncated to 0
#         plot_quantity = np.repeat(bins[:-1], (hist * 1000).astype(int))

#         current_kwargs = kwargs.copy()
        
#         if label:
#             print(f"Histogram for: {label} at parameter {p}")
#         else:
#             print(f"Histogram at parameter {p}")

#         uplt.histogram(plot_quantity, **current_kwargs)

#     if info_text:
#         print(info_text)

    return None, None

if __name__ == '__main__':

    data = np.random.randn(32, 64)

    fig, ax = hist(data, backend='uniplot')

    plt.show()