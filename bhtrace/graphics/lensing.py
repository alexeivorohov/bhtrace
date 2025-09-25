from typing import Dict, Tuple, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from bhtrace.trajectory.trajectory import Trajectory

import torch
import numpy as np
import matplotlib.pyplot as plt

from . import Plot2D, opt_mosaic, Coloring

class LensingPlot:
    """
    Provides methods for creating lensing plots (deflection angle vs. impact parameter).
    """
    @classmethod
    def plot(cls,
             dphi: torch.Tensor | Dict[str, torch.Tensor] = None,
             b: torch.Tensor | Dict[str, torch.Tensor] = None,
             traj: 'Trajectory' | Dict[str, 'Trajectory'] = None,
             add_2dplot: bool = False,
             fig: plt.Figure = None):
        """
        Plots lensing data. Can also show the 2D trajectory plot.
        Data can be provided as 'dphi' and 'b' tensors/dicts, or extracted from 'traj' object(s).
        """
        if traj is not None:
            dphi, b = cls._extract_from_traj_or_use_provided(dphi, b, traj)

        if dphi is None or b is None:
            raise ValueError("LensingPlot.plot requires either 'traj' or both 'dphi' and 'b'.")

        if not isinstance(dphi, dict):
            dphi = {'data': dphi}
            b = {'data': b}

        axes_dict = cls._setup_axes(add_2dplot, fig)
        lensing_ax = axes_dict['lensing']

        dataset_colors = plt.cm.viridis(np.linspace(0, 1, len(dphi)))

        for i, key in enumerate(dphi.keys()):
            cls.plot1(b[key], dphi[key], lensing_ax, dataset_color=dataset_colors[i], label=key)

        lensing_ax.set_xlabel("Impact parameter b")
        lensing_ax.set_ylabel("Number of windings n")
        lensing_ax.grid(True)
        if len(dphi) > 1:
            lensing_ax.legend()

        if add_2dplot:
            if traj is None:
                print("Warning: add_2dplot=True but no trajectory data provided.")
            else:
                plot_2d_ax = axes_dict['2d_plot']
                trajs_to_plot = list(traj.values()) if isinstance(traj, dict) else [traj]
                labels = list(traj.keys()) if isinstance(traj, dict) else None
                Plot2D.plot(trajs_to_plot, ax=plot_2d_ax, labels=labels)
                plot_2d_ax.set_title("Trajectories")

        plt.tight_layout()
        plt.show()
        return plt.gcf(), axes_dict

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
        return axes

    @classmethod
    def plot1(cls, b: torch.Tensor, dphi: torch.Tensor, ax: plt.Axes, dataset_color=None, label: str = None):
        """
        Helper to plot one set of (b, dphi) data, colored by winding number.
        """
        n = dphi / (2 * torch.pi)
        
        sort_indices = torch.argsort(b)
        b_sorted, n_sorted = b[sort_indices].cpu(), n[sort_indices].cpu()

        colors = Coloring.lensing_colors(n_sorted)
        
        ax.scatter(b_sorted, n_sorted, c=colors, label=label if label != 'data' else None, s=10, zorder=2)
        ax.plot(b_sorted, n_sorted, color=dataset_color, alpha=0.5, zorder=1)

    @classmethod
    def __from_traj__(cls, traj: 'Trajectory') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts deflection angle (dphi) and impact parameter (b) from a Trajectory.
        Assumes photon in stationary, axisymmetric spacetime, with b = |p_phi / p_t|.
        """
        phi = traj.x[..., 3]
        dphi = phi[..., -1] - phi[..., 0]

        if not hasattr(traj, 'p'):
            raise AttributeError("Trajectory needs 'p' attribute to calculate impact parameter.")
        
        p_phi, p_t = traj.p[..., 0, 3], traj.p[..., 0, 0]
        
        b = torch.zeros_like(p_t)
        non_zero_pt = torch.abs(p_t) > 1e-9
        b[non_zero_pt] = torch.abs(p_phi[non_zero_pt] / p_t[non_zero_pt])
        
        return dphi, b