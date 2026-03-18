from typing import Dict, Tuple, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from bhtrace.trajectory.trajectory import Trajectory

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from . import Plot2D, opt_mosaic, Coloring

class LensingPlot:
    """
    Provides methods for creating lensing plots (deflection angle vs. impact parameter).
    """
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


    @classmethod
    def plot(cls,
             dphi: torch.Tensor | Dict[str, torch.Tensor] = None,
             b: torch.Tensor | Dict[str, torch.Tensor] = None,
             traj: 'Trajectory' | Dict[str, 'Trajectory'] = None,
             e_b: torch.Tensor | Dict[str, torch.Tensor ] = None,
             show_peaks: bool = True,
             show_points: bool = False,
             add_2dplot: bool = False,
             peak_dist: int = 2,
             fig: plt.Figure = None
            ):
        """
        Plots lensing data. Can also show the 2D trajectory plot.
        Data can be provided as 'dphi' and 'b' tensors/dicts, or extracted from 'traj' object(s).

        Call examples:
            Lensing.plot(dphi, b)
            Lensing.plot(traj=traj)
            Lensing.plot(traj=traj, add_2dplot=True)
        
        Args:
            dphi: torch.Tensor | Dict[torch.Tensors] - inclination angles. 
            If dict passed, lensing curves will be displayed on the same axes and labeled with keys of this dict.
            b: torch.Tensor | Dict[torch.Tensors] - impact factors
            For dict behaviour is same as for dphi.
            traj: Trajectory | Dict[Trajectory] - trajectories.
            Will try to use .lens attribute to extract dphi and b, if present.
            e_b: torch.Tensor | Dict[Trajectory] - unit vector in the direction of impact factor increase.
            show_peaks: bool - determine and note peaks of lensing function (default: true)
            add_2dplot: bool - add trajectories 2d plot (default: False)

        """
        if traj is not None:
            dphi, b = cls._extract_from_traj_or_use_provided(dphi, b, traj)

        if dphi is None or b is None:
            raise ValueError("LensingPlot.plot requires either 'traj' or both 'dphi' and 'b'.")

        if not isinstance(dphi, dict):
            dphi = {'data': dphi}
            b = {'data': b}

        if e_b is None:
            e_b = [0., 1., 0.]

        axes_dict, fig = cls._setup_axes(add_2dplot, fig)
        lensing_ax = axes_dict['lensing']

        dataset_colors = plt.cm.viridis(np.linspace(0, 1, len(dphi)))

        # Plotting
        for i, key in enumerate(dphi.keys()):
            cls.plot1(b[key],
                      dphi[key],
                      lensing_ax,
                      dataset_color=dataset_colors[i],
                      label=key,
                      show_peaks=show_peaks,
                      show_points=show_points,
                      peak_dist=peak_dist
                      )

        lensing_ax.set_xlabel("Impact parameter b")
        lensing_ax.set_ylabel("Number of windings n")
        lensing_ax.grid(True)
        if len(dphi) > 1:
            lensing_ax.legend()

        # 2d plot
        if add_2dplot:
            if traj is None:
                print("Warning: add_2dplot=True but no trajectory data provided.")
            else:
                plot_2d_ax = axes_dict['2d_plot']
                trajs_to_plot = list(traj.values()) if isinstance(traj, dict) else [traj]
                labels = list(traj.keys()) if isinstance(traj, dict) else None
                Plot2D.plot(trajs_to_plot, ax=plot_2d_ax, labels=labels)
                plot_2d_ax.set_title("Trajectories")

        return axes_dict, fig

    @classmethod
    def _extract_from_traj_or_use_provided(cls, dphi, b, traj):
        if isinstance(traj, dict):
            if dphi is None: dphi = {}
            if b is None: b = {}
            for key, t in traj.items():
                dphi_from_traj, b_from_traj = cls.__from_traj__(t)
                if key not in dphi: dphi[key] = dphi_from_traj
                if key not in b: b[key] = b_from_traj
        else: # Single trajectory
            dphi_from_traj, b_from_traj = cls.__from_traj__(traj)
            if dphi is None: dphi = dphi_from_traj
            if b is None: b = b_from_traj
        return dphi, b

    @classmethod
    def _setup_axes(cls, add_2dplot, fig):
        ax_ids = ['lensing']
        if add_2dplot:
            ax_ids.append('2d_plot')
        
        shape, mosaic = opt_mosaic(ax_ids)
        
        if fig is None:
            fig, axes = plt.subplot_mosaic(mosaic, figsize=(8 * shape[1], 6 * shape[0]))
        else:
            axes = fig.subplot_mosaic(mosaic)
        return axes, fig

    @classmethod
    def plot1(cls,
              b: torch.Tensor,
              dphi: torch.Tensor, 
              ax: plt.Axes,
              dataset_color=None,
              label: str = None,
              show_points = False,
              show_peaks = True,
              peak_dist = 2
              ):
        """
        Helper to plot one set of (b, dphi) data, colored by winding number.

        Requires b to be incremental. 
        """

        n = dphi / (2 * torch.pi)

        # colors = Coloring.lensing_colors(n)
        
        label=label if label != 'data' else None
        if show_points:
            ax.scatter(b, n, c=dataset_color, **cls.__curve_p_style__)
        ax.plot(b, n, color=dataset_color, label=label, **cls.__curve_style__)
        y_bottom, y_top = ax.get_ylim()
        if show_peaks:
            peaks, properties = find_peaks(n.numpy(), distance = peak_dist)
            for peak in peaks:
                ax.plot([b[peak], b[peak]], [0, y_top], c=dataset_color, **cls.__peak_style__)
                # ax.axline()
            ax.scatter(b[peak], n[peak], color=dataset_color, **cls.__peak_p_style__)
            print(peaks)
            print('Case {label}')
            print(properties)

    @classmethod
    def __from_traj__(cls, traj: 'Trajectory', e_b = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts deflection angle (dphi) and impact parameter (b) from a Trajectory.
        
        Will try to use traj.lens property, if not presnent, eval_lens will be called;
        """
        if hasattr(traj, 'lens'):
            return traj.lens

        from bhtrace.scenarios.lensing import eval_lens
        # Refactor
        b = traj.x[..., 2]
        dphi = eval_lens(traj=traj)
     
        return dphi, b