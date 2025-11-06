'''
3D plotting utilities.
'''
from typing import Tuple, List, Optional

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

from .presets import Coloring


class Plot3D:
    '''
    A utility class for creating common 3D plots using Matplotlib.

    This class provides a set of static methods for generating various types of 3D visualizations,
    such as point clouds, lines, and vector fields. It simplifies the process of creating
    and customizing these plots by providing a consistent interface and handling the setup
    of figures and axes.
    '''

    default_figsize: Tuple[int, int] = (10, 10)
    '''Default figure size for 3D plots.'''
    default_direction: torch.Tensor = torch.Tensor([-1, -1, -1])
    '''Default view direction for 3D plots.'''

    @classmethod
    def fig_handler(
            cls,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
        ):
        '''
        Handles figure and axes creation for plots.

        This method allows for flexible plot creation by handling different scenarios:
        - If no fig or ax are passed, a new figure and 3D axes are created with default parameters.
        - If a fig is provided, a new 3D axes is added to this figure.
        - If an ax is provided, it is used for plotting.

        Args:
            fig (plt.Figure, optional): The figure to which the plot will be added. Defaults to None.
            ax (plt.Axes, optional): The axes on which the plot will be drawn. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the figure and axes for the plot.
        '''
        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=cls.default_figsize)
            ax = fig.add_subplot(projection='3d')
        else:
            fig = ax.get_figure()

        return fig, ax

    @classmethod
    def dir_handler(
        cls,
        ax: plt.Axes,
        dir = None,
        elev: float = None,
        azim: float = None,
        roll: float = None,
        ):
        '''
        Handles the viewing direction of the 3D plot.

        This method is intended to allow setting the camera direction using either a direction vector
        or Euler angles (elevation, azimuth, roll).

        Args:
            ax (plt.Axes): The axes to apply the view direction to.
            dir (torch.Tensor, optional): A vector specifying the viewing direction. Defaults to None.
            elev (float, optional): The elevation angle in degrees. Defaults to None.
            azim (float, optional): The azimuth angle in degrees. Defaults to None.
            roll (float, optional): The roll angle in degrees. Defaults to None.
        '''
        if dir is not None:
            raise NotImplementedError("Direction vector not implemented yet.")
        ax.view_init(elev=elev, azim=azim, roll=roll)

    @classmethod
    def cmap_handler(
        cls,
        values: torch.Tensor,
        existing_cmap: Optional[mpl.colors.Colormap] = None,
        ):
        '''
        Handles colormaps for plot primitives.

        For a given set of values, this method returns a colormap, a normalization object,
        and the mapped colors.

        Args:
            values (torch.Tensor): The values to be mapped to colors.
            existing_cmap (mpl.colors.Colormap, optional): An existing colormap to use.
                Defaults to 'viridis'.

        Returns:
            Tuple[torch.Tensor, mpl.colors.Colormap, mpl.colors.Normalize]: A tuple containing
                the RGBA colors, the colormap, and the normalization object.
        '''
        if existing_cmap is None:
            cmap = plt.get_cmap('viridis')
        else:
            cmap = existing_cmap
        
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        colors = cmap(norm(values.numpy()))
        return torch.from_numpy(colors), cmap, norm

    @classmethod
    def point_cloud(
            cls,
            points: torch.Tensor,
            values: Optional[torch.Tensor] = None,
            dir: Optional[torch.Tensor] = None,
            elev: Optional[float] = None,
            azim: Optional[float] = None,
            roll: Optional[float] = None,
            fig: Optional[plt.Figure] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs
        ):
        """
        Plots a 3D point cloud.

        Args:
            points (torch.Tensor): Points to draw. The last dimension is treated as coordinates.
                Shape: [..., 3].
            values (torch.Tensor, optional): Scalar values for each point, used for coloring.
                Defaults to None.
            dir (torch.Tensor, optional): Viewing direction vector. Defaults to None.
            elev (float, optional): Elevation angle of the camera. Defaults to None.
            azim (float, optional): Azimuth angle of the camera. Defaults to None.
            roll (float, optional): Roll angle of the camera. Defaults to None.
            fig (plt.Figure, optional): Figure to plot on. If None, a new one is created.
                Defaults to None.
            ax (plt.Axes, optional): Axes to plot on. If None, a new one is created.
                Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes of the plot.
        """
        fig, ax = cls.fig_handler(fig, ax)
        cls.dir_handler(ax, dir=dir, elev=elev, azim=azim, roll=roll)

        x = points[..., 0].view(-1)
        y = points[..., 1].view(-1)
        z = points[..., 2].view(-1)

        c = None
        if values is not None:
            c = values.view(-1)

        ax.scatter(x, y, z, c=c, marker='.', **kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return fig, ax

    @classmethod
    def lines(
        cls,
        points: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        dir: Optional[torch.Tensor] = None,
        elev: Optional[float] = None,
        azim: Optional[float] = None,
        roll: Optional[float] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
        ):
        '''
        Plots 3D parametric lines.

        This method can plot batches of lines. The input tensor `points` is expected
        to have the time/parameter dimension as the second to last, and the coordinate
        dimension as the last.

        Args:
            points (torch.Tensor): A tensor of shape [..., T, 3] representing points on the lines.
                Axis -1 is for coordinates (X, Y, Z), axis -2 is the parameter (e.g., time).
                Other axes are treated as batch dimensions.
            values (torch.Tensor, optional): Scalar values for coloring. Assumed to be per-line.
                Defaults to None.
            dir (torch.Tensor, optional): Viewing direction vector. Defaults to None.
            elev (float, optional): Elevation angle of the camera. Defaults to None.
            azim (float, optional): Azimuth angle of the camera. Defaults to None.
            roll (float, optional): Roll angle of the camera. Defaults to None.
            fig (plt.Figure, optional): Figure to plot on. If None, a new one is created.
                Defaults to None.
            ax (plt.Axes, optional): Axes to plot on. If None, a new one is created.
                Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes of the plot.
        '''

        fig, ax = cls.fig_handler(fig, ax)
        cls.dir_handler(ax, dir=dir, elev=elev, azim=azim, roll=roll)

        points_flat = points.view(-1, points.shape[-2], points.shape[-1])

        if values is None:
            for p in points_flat:
                ax.plot(p[..., 0], p[..., 1], p[..., 2], **kwargs)
        else:
            colors, _, _ = cls.cmap_handler(values.view(-1))
            for i, p in enumerate(points_flat):
                ax.plot(p[..., 0], p[..., 1], p[..., 2], color=colors[i].numpy(), **kwargs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return fig, ax

    @classmethod
    def vector_field(
        cls,
        points: torch.Tensor,
        vectors: torch.Tensor,
        values: torch.Tensor = None,
        dir: torch.Tensor = None,
        elev: Optional[float] = None,
        azim: Optional[float] = None,
        roll: Optional[float] = None,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        **kwargs
        ):
        '''
        Plots a 3D vector field.

        This method visualizes a vector field by drawing arrows (quivers) at specified points.

        Args:
            points (torch.Tensor): A tensor of shape [..., 3] for the origins of the vectors.
            vectors (torch.Tensor): A tensor of shape [..., 3] for the vector components.
            values (torch.Tensor, optional): Scalar values for coloring the vectors. Defaults to None.
            dir (torch.Tensor, optional): Viewing direction vector. Defaults to None.
            elev (float, optional): Elevation angle of the camera. Defaults to None.
            azim (float, optional): Azimuth angle of the camera. Defaults to None.
            roll (float, optional): Roll angle of the camera. Defaults to None.
            fig (plt.Figure, optional): Figure to plot on. If None, a new one is created.
                Defaults to None.
            ax (plt.Axes, optional): Axes to plot on. If None, a new one is created.
                Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes of the plot.
        '''

        fig, ax = cls.point_cloud(points, values=None, dir=dir, elev=elev, azim=azim, roll=roll, fig=fig, ax=ax, **kwargs)

        x = points[..., 0].view(-1)
        y = points[..., 1].view(-1)
        z = points[..., 2].view(-1)

        u_x = vectors[..., 0].view(-1)
        u_y = vectors[..., 1].view(-1)
        u_z = vectors[..., 2].view(-1)

        q = ax.quiver(x, y, z, u_x, u_y, u_z, **kwargs)

        if values is not None:
            colors, _, _ = cls.cmap_handler(values.view(-1))
            q.set_color(colors.numpy())


        return fig, ax
