from typing import Dict, Tuple, Iterable, TYPE_CHECKING

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if TYPE_CHECKING:
    from bhtrace.trajectory.trajectory import Trajectory

from .presets import opt_mosaic

class Plot2D:
    """
    Provides methods for 2D plotting of trajectories.
    """

    @classmethod
    def plot(cls, trajs: Iterable['Trajectory'], ax: plt.Axes = None, colors: list = None, labels: list = None):
        """
        Plots one or more trajectories in a 2D plane.
        By default, it plots the coordinates at index 1 and 2 of the trajectory data.

        Args:
            trajs (Iterable[Trajectory]): A single Trajectory object or an iterable of them.
            ax (plt.Axes, optional): A matplotlib axes object to plot on. If None, creates a new one.
            colors (list, optional): A list of colors to use for the trajectories.
            labels (list, optional): A list of labels for the trajectories.

        Returns:
            plt.Axes: The axes object with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if not isinstance(trajs, (list, tuple)):
            trajs = [trajs]

        if colors is None:
            colors = Coloring.trajectory_colors(len(trajs))

        for i, traj in enumerate(trajs):
            coords, _ = traj['Cartesian']
            coords = coords.cpu()
            label = labels[i] if labels and i < len(labels) else f'traj_{i}'
            ax.plot(coords[..., 1], coords[..., 2], color=colors[i], label=label)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal', 'box')
        ax.grid(True)
        if labels:
            ax.legend()
        return fig, ax
    

    @classmethod
    def plot_2d(cls, traj: 'Trajectory', ax: plt.Axes = None, figsize=(10, 10), **kwargs):
        """
        Plots a 2D projection of a single trajectory.
        Includes a circle representing the black hole event horizon.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        X, _ = traj['Cartesian']
        
        # Plot black hole horizon (assuming r=2M)
        circle = patches.Circle((0, 0), 2.0, edgecolor='black', facecolor='black', lw=2)
        ax.add_patch(circle)

        ax.plot(X[..., 1].cpu().numpy().T, X[..., 2].cpu().numpy().T, **kwargs)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-10, 20])
        ax.set_ylim([-15, 15])
        ax.grid('on')
        ax.set_xlabel('$Y/M$')
        ax.set_ylabel('$Z/M$')
        return fig, ax
    

    @classmethod
    def plot_2d_mosaic(cls, trajectories: Iterable['Trajectory'], figsize=(10, 10), **kwargs):
        """
        Plots multiple 2D trajectories on a mosaic of subplots.
        """
        if not isinstance(trajectories, (list, tuple)):
            trajectories = [trajectories]
            
        traj_keys = [str(i) for i in range(len(trajectories))]
        shape, mosaic = opt_mosaic(traj_keys)
        figsize_ = (shape[0] * figsize[0], shape[1] * figsize[1])

        fig, axs = plt.subplot_mosaic(mosaic, figsize=figsize_)

        for k, traj in zip(traj_keys, trajectories):
            ax = axs[k]
            cls.plot_2d(traj, ax=ax, **kwargs)
            ax.set_title(k)

        return fig