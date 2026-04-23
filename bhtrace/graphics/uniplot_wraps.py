"""This file wraps the methods of Uniplot package in object-oriented way"""

import uniplot as uplt
import numpy as np
from typing import List, Optional, Any, Dict, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from bhtrace.graphics.uniplot_wraps import UniFigure

# TODO: fix two strange tests
# TODO: describe common kwargs or expose new args
# TODO: histogram ridge plots
# TODO: colorbars

class UniAxes:
    """
    An object that holds information about a single plot.
    Matplotlib-like axes.
    """

    def __init__(self, figure: "UniFigure"):
        self.figure = figure
        self.series_ys: List[np.ndarray] = []
        self.series_xs: List[Optional[np.ndarray]] = []
        self.series_kwargs: List[Dict[str, Any]] = []
        self.plot_options: Dict[str, Any] = {}

    def plot(self, ys: np.ndarray, xs: Optional[np.ndarray] = None, **kwargs):
        """Add a line plot series."""
        self.series_ys.append(ys)
        self.series_xs.append(xs)
        self.series_kwargs.append(kwargs)

    def scatter(self, ys: np.ndarray, xs: Optional[np.ndarray] = None, **kwargs):
        """Add a scatter plot series."""
        kwargs["lines"] = False
        self.series_ys.append(ys)
        self.series_xs.append(xs)
        self.series_kwargs.append(kwargs)

    def stairs(self, values: np.ndarray, edges: np.ndarray, **kwargs):
        """Add a stairs plot (matplotlib.pyplot.stairs analog)."""
        if len(edges) != len(values) + 1:
            raise ValueError(
                f"edges length must be values length + 1, but {len(edges)} != {len(values)} + 1"
            )
        xs = np.repeat(edges, 2)[1:-1]
        ys = np.repeat(values, 2)
        self.plot(ys, xs, **kwargs)

    def bar(
        self,
        x: np.ndarray,
        height: np.ndarray,
        width: Union[float, np.ndarray] = 0.8,
        **kwargs,
    ):
        """Add a bar plot (matplotlib.pyplot.bar analog)."""
        if isinstance(width, (int, float)):
            width = np.full_like(x, width, dtype=float)

        xs = []
        ys = []
        for i in range(len(x)):
            x_left = x[i] - width[i] / 2
            x_right = x[i] + width[i] / 2
            h = height[i]
            xs.extend([x_left, x_left, x_right, x_right])
            ys.extend([0, h, h, 0])

        self.plot(np.array(ys), np.array(xs), **kwargs)

    def histogram(
        self, 
        x: np.ndarray, 
        bins: int | np.ndarray = 10, 
        density: bool = False, 
        **kwargs
    ):
        """Plots a histogram"""
        counts, bin_edges = np.histogram(x, bins=bins, density=density)
        self.stairs(counts, bin_edges, **kwargs)

    def set_title(self, title: str):
        """Set the plot title."""
        self.plot_options["title"] = title

    def set_xlabel(self, label: str):
        """Set the x-axis label."""
        self.plot_options["x_label"] = label

    def set_ylabel(self, label: str):
        """Set the y-axis label."""
        self.plot_options["y_label"] = label

    def set_xlim(self, lim: Tuple[float, float]):
        """Set the x-axis limits."""
        self.plot_options["x_min"] = lim[0]
        self.plot_options["x_max"] = lim[1]

    def set_ylim(self, lim: Tuple[float, float]):
        """Set the y-axis limits."""
        self.plot_options["y_min"] = lim[0]
        self.plot_options["y_max"] = lim[1]


class UniFigure:
    """
    A container for one or more UniAxes objects.
    Matplotlib-like figure.
    """

    def __init__(self, **kwargs):
        self.axes: List[UniAxes] = []
        self.kwargs = kwargs

    def add_subplot(self, *args, **kwargs) -> UniAxes:
        """Adds a subplot (UniAxes) to the figure."""
        if self.axes:
            return self.axes[0]
        ax = UniAxes(self)
        self.axes.append(ax)
        return ax

    def show(self):
        """Render the plot(s) to the terminal."""
        for ax in self.axes:
            if not ax.series_ys:
                continue

            lines_opts = [s.get("lines", True) for s in ax.series_kwargs]
            if len(set(lines_opts)) > 1:
                raise ValueError(
                    "Uniplot wrapper cannot mix line and scatter plots on the same axes."
                )

            plot_kwargs = ax.plot_options.copy()
            if lines_opts:
                plot_kwargs["lines"] = lines_opts[0]

            legend_labels = [s.get("label") for s in ax.series_kwargs]
            if any(l is not None for l in legend_labels):
                plot_kwargs["legend_labels"] = [
                    l for l in legend_labels if l is not None
                ]

            xs_to_plot = []
            has_any_xs = any(xs is not None for xs in ax.series_xs)

            if has_any_xs:
                for i, xs in enumerate(ax.series_xs):
                    if xs is not None:
                        xs_to_plot.append(xs)
                    else:
                        xs_to_plot.append(np.arange(len(ax.series_ys[i])))
                plot_kwargs["xs"] = xs_to_plot

            uplt.plot(ys=ax.series_ys, **plot_kwargs)


def subplots(
    nrows: int = 1, ncols: int = 1, **kwargs
) -> Tuple["UniFigure", "UniAxes"]:
    """
    Create a figure and a set of subplots.
    This function is a wrapper around UniFigure and UniAxes to provide a
    matplotlib-like interface.
    """
    if nrows != 1 or ncols != 1:
        raise NotImplementedError("uniplot wrapper only supports single plots for now.")

    fig = UniFigure(**kwargs)
    ax = fig.add_subplot(111)
    return fig, ax


if __name__ == '__main__':

    import numpy as np

    # x = np.linspace(0, 6, 128)
    # y_1 = np.sin(x)
    # y_2 = np.cos(x)

    # fig, ax = subplots()

    # ax.plot(y_1, x)
    # ax.plot(y_2, x)

    # fig.show() 
    
    
    fig, ax = subplots()

    bins = np.linspace(-5, 5, 18)

    dist_1 = np.random.randn(1024)
    dist_2 = np.random.randn(128)
    
    ax.histogram(dist_1, bins=bins, density=True)
    # ax.histogram(dist_2)

    fig.show()
