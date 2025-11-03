'''

'''

from .presets import Coloring

import torch
import matplotlib.pyplot as plt

class Plot3D:

    figsize = (10, 10)

    @classmethod
    def fig_handler(
            cls,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
        ):
        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=cls.figsize)
            ax = fig.add_subplot(projection='3d')
        else:
            fig = ax.get_figure()

        return fig, ax

    @classmethod
    def angle_handler(
        cls,
        elev: float = None,
        azim: float = None,
        roll: float = None,
        ):
        
        return None

    @classmethod
    def point_cloud(
            cls,
            points: torch.Tensor,
            values: torch.Tensor = None,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            dir: torch.Tensor = None,
            roll: float = 0,
            elev: float = 0,
            azim: float = None,
        ):
        """

        Args:
            - points: torch.Tensor - points to draw. The last dimension 
            - values: torch.Tensor (default: None)
            - dir: torch.Tensor ()
            - fig: plt.Figure
            - ax: plt.Axes
        """
        fig, ax = cls.fig_handler(fig, ax)
        # elev, azim = 
        if values:
            # c = Coloring.point_colormap()
            pass
        # for i, pos in enumerate(net.pos3d):
        #     ax2.text(*pos, f'{i}')
        x = points[..., 0].view(-1)
        y = points[..., 1].view(-1)
        z = points[..., 2].view(-1)
        ax.scatter(x, y, z, marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.view_init()

        return fig, ax

    @classmethod
    def vector_field(
        cls,
        points: torch.Tensor,
        vectors: torch.Tensor,
        values: torch.Tensor = None,
        dir: torch.Tensor = None,
        fig: plt.Figure = None,
        ax: plt.Axes = None
        ):

        fig, ax = cls.point_cloud(points, values, dir, fig, ax)

        # quiver

        return fig, ax