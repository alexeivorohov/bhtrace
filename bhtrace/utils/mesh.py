"""
This module provides classes for generating and managing coordinate meshes.

Meshes are primarily used to define the initial conditions for tracing
trajectories, for instance, by defining a grid of starting points for photons
on an observer's screen.

The module includes a base `Mesh` class and implementations for various shapes
such as `Linear`, `Rectangle`, and adaptively refined `Nonuniform` meshes.
"""

from abc import ABC, abstractmethod
import math
from typing import Tuple, List, Callable, Optional

import torch
import bhtrace.graphics as bhg


class Mesh(ABC):
    """Abstract base class for all coordinate mesh implementations.

    This class defines the common interface and attributes for all meshes,
    such as node positions, weights, and cell structures.

    Attributes:
        pos3d (torch.Tensor): A tensor of shape (N, 3) holding the Cartesian coordinates of the N nodes.
        cells (List[List[int]]): A list of cells, where each cell is a list of node indices.
        gen (int): The number of upsampling generations this mesh has undergone.
        uniform (bool): A flag indicating if the mesh has uniform spacing.
        N (int): The total number of nodes in the mesh.
        anchor (torch.Tensor): The anchor point of the mesh, used for positioning or rotation.
        weights (torch.Tensor): A tensor of shape (N,) representing the area weight of each node.
        traced (torch.Tensor): A boolean tensor of shape (N,) used as a mask to indicate which nodes have been traced.
        generation (List[int]): A list of length N indicating the upsampling generation of each node.
    """

    pos3d: torch.Tensor = None
    gen: int = 0
    """Number of refinement steps done"""

    """Weights of each vertex, must sum up to 1"""
    uniform: bool = True

    cells: List[List[int]] = None
    active_cells: list[int]
    traced: List[int]

    def __init__(
        self,
        anchor: torch.Tensor = torch.zeros(3),
        pos3d: torch.Tensor = None,
        cells: torch.Tensor = None,
        weights: Optional[torch.Tensor] = None,
        traced: Optional[torch.Tensor] = None,
        generation: Optional[List[int]] = None,
    ):
        self.anchor = anchor
        self.pos3d = pos3d
        self.cells = cells
        self.weights = weights
        self.traced = traced
        self.generation = generation
        self.N = pos3d.shape[0] if pos3d is not None else 0

    @abstractmethod
    def info(self):
        """Prints information about the mesh to the console."""
        pass

    def to(self, device=None, dtype=None):
        """Moves and/or casts the tensors of the mesh.

        This method iterates over all tensor attributes of the mesh and moves them
        to the specified device. For floating-point tensors, it also casts them
        to the specified dtype.

        Args:
            device (str, optional): The device to move the tensors to.
            dtype (torch.dtype, optional): The data type to cast floating-point tensors to.

        Returns:
            mesh: The modified mesh object.
        """
        for attr_name in dir(self):
            if attr_name.startswith("__"):
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                if attr.is_floating_point() and dtype is not None:
                    attr = attr.to(device=device, dtype=dtype)
                else:
                    attr = attr.to(device=device)
                setattr(self, attr_name, attr)
        return self

    def show(self, direction=None, fig=None, ax=None):
        """Visualizes the mesh as a 3D point cloud.

        Args:
            direction: (Not yet implemented).
            fig: An existing matplotlib figure.
            ax: An existing matplotlib 3D axes.

        Returns:
            A tuple containing the figure and axes.
        """
        return bhg.plot3d.point_cloud(self.pos3d, self.generation, fig=fig, ax=ax)




class Linear(Mesh):
    """A 1D, linear arrangement of nodes.

    Creates a straight line of `npoints` nodes centered at `X0` and oriented along the `size` vector.
    """

    def __init__(
        self,
        size: torch.Tensor,
        npoints: int,
        X0: torch.Tensor = torch.zeros(3),
        anchor: torch.Tensor = torch.zeros(3),
    ):
        pos3d = X0 + torch.linspace(-0.5, 0.5, npoints).unsqueeze(-1) * size.unsqueeze(
            0
        )
        cells = [[i, i + 1] for i in range(npoints - 1)]
        weights = torch.ones(npoints)
        traced = torch.zeros(npoints, dtype=torch.bool)
        generation = [0] * npoints

        super().__init__(anchor, pos3d, cells, weights, traced, generation)
        self.shape = (npoints,)
        self.size = size
        self.X0 = X0

    def info(self):
        """Prints information about the Linear mesh."""
        print(f"Linear mesh, shape: {self.shape}, X0: {self.X0}")


class Rectangle(Mesh):
    """A 2D, rectangular grid of nodes.

    Args:
        size (torch.Tensor): A 2-element tensor defining the width and height of the rectangle.
        shape (Tuple[int, int]): The number of nodes along the y and z dimensions.
        X0 (torch.Tensor): The 3D coordinate of the rectangle's center.
        anchor (torch.Tensor): The anchor point of the mesh.
    """

    def __init__(
        self,
        size: torch.Tensor,
        shape: Tuple[int, int] = (5, 5),
        X0: torch.Tensor = torch.zeros(3),
        anchor: torch.Tensor = torch.zeros(3),
    ):
        y_ = torch.linspace(-0.5, 0.5, shape[0]) * size[0]
        z_ = torch.linspace(-0.5, 0.5, shape[1]) * size[1]
        y_grid, z_grid = torch.meshgrid(y_, z_, indexing="ij")
        x_grid = torch.zeros_like(y_grid)

        pos3d = X0 + torch.stack((x_grid, y_grid, z_grid), dim=-1)
        pos3d = pos3d.flatten(0, -2)
        cells = [
            [
                k * shape[1] + i,
                k * shape[1] + i + 1,
                (k + 1) * shape[1] + i + 1,
                (k + 1) * shape[1] + i,
            ]
            for i in range(shape[1] - 1)
            for k in range(shape[0] - 1)
        ]
        n_points = shape[0] * shape[1]
        weights = torch.ones(n_points)
        traced = torch.zeros(n_points, dtype=torch.bool)
        generation = [0] * n_points

        super().__init__(anchor, pos3d, cells, weights, traced, generation)
        self.size = size
        self.shape = shape
        self.X0 = X0

    def info(self):
        """Prints information about the Rectangle mesh."""
        print(f"Rectangle mesh, shape: {self.shape}, X0: {self.X0}")


class Hex(Mesh):
    """A 2D, hexagonal grid of nodes.

    This implementation uses axial coordinates to generate a "pointy-top" hexagonal
    grid arranged in a parallelogram shape.

    Args:
        size (torch.Tensor): A 2-element tensor defining the approximate width and height of the grid.
        shape (Tuple[int, int]): The number of columns (q) and rows (r) in the axial coordinate system.
        X0 (torch.Tensor): The 3D coordinate of the grid's center.
        anchor (torch.Tensor): The anchor point of the mesh.
    """

    def __init__(
        self,
        size: torch.Tensor,
        shape: Tuple[int, int] = (5, 5),
        X0: torch.Tensor = torch.zeros(3),
        anchor: torch.Tensor = torch.zeros(3),
    ):
        self.size = size
        n_q, n_r = shape
        N = n_q * n_r

        hex_size = self.size[0] / (1.5 * n_q)

        positions = []
        for q in range(n_q):
            for r in range(n_r):
                y = hex_size * 1.5 * q
                z = hex_size * math.sqrt(3) * (r + q / 2.0)
                positions.append([0, y, z])

        pos3d = torch.tensor(positions, dtype=torch.float32)

        if N > 0:
            mean_pos = pos3d.mean(dim=0)
            pos3d -= mean_pos

        pos3d += X0

        cells = []
        for q in range(n_q - 1):
            for r in range(n_r - 1):
                p1 = q * n_r + r
                p2 = (q + 1) * n_r + r
                p3 = q * n_r + (r + 1)
                p4 = (q + 1) * n_r + (r + 1)

                cells.append([p1, p2, p3])
                cells.append([p2, p4, p3])

        weights = torch.ones(N)
        traced = torch.zeros(N, dtype=torch.bool)
        generation = [0] * N

        super().__init__(anchor, pos3d, cells, weights, traced, generation)
        self.size = size
        self.shape = shape
        self.X0 = X0

    def info(self):
        """Prints information about the Hex mesh."""
        print(f"Hex mesh, shape: {self.shape}, X0: {self.X0}")


class Circle(Mesh):
    """A 2D, polar grid of nodes.

    The grid is composed of a central point, surrounded by concentric rings.

    Args:
        size (torch.Tensor): A 1-element tensor defining the diameter of the circle.
        shape (Tuple[int, int]): The number of rings and the number of wedges (radial divisions).
        X0 (torch.Tensor): The 3D coordinate of the circle's center.
        anchor (torch.Tensor): The anchor point of the mesh.
    """

    def __init__(
        self,
        size: torch.Tensor,
        shape: Tuple[int, int] = (5, 5),
        X0: torch.Tensor = torch.zeros(3),
        anchor: torch.Tensor = torch.zeros(3),
    ):
        self.size = size
        n_rings, n_wedges = shape
        N = 1 + n_rings * n_wedges

        radius = self.size[0] / 2.0

        center_point = torch.tensor([[0.0, 0.0, 0.0]])

        if n_rings > 0:
            radii = torch.linspace(radius / n_rings, radius, n_rings)
            angles = torch.linspace(0, 2 * torch.pi, n_wedges + 1)[:-1]
            r_grid, a_grid = torch.meshgrid(radii, angles, indexing="ij")

            y = r_grid * torch.cos(a_grid)
            z = r_grid * torch.sin(a_grid)
            x = torch.zeros_like(y)

            ring_points = torch.stack((x, y, z), dim=-1).flatten(0, -2)
            pos3d = X0 + torch.cat((center_point, ring_points), dim=0)
        else:
            pos3d = X0 + center_point

        cells = []
        if n_rings > 0:
            for j in range(n_wedges):
                p1 = 1 + j
                p2 = 1 + (j + 1) % n_wedges
                cells.append([0, p1, p2])

            for i in range(n_rings - 1):
                for j in range(n_wedges):
                    p1 = 1 + i * n_wedges + j
                    p2 = 1 + i * n_wedges + (j + 1) % n_wedges
                    p3 = 1 + (i + 1) * n_wedges + (j + 1) % n_wedges
                    p4 = 1 + (i + 1) * n_wedges + j
                    cells.append([p1, p2, p3, p4])

        weights = torch.ones(N)
        traced = torch.zeros(N, dtype=torch.bool)
        generation = [0] * N

        super().__init__(anchor, pos3d, cells, weights, traced, generation)
        self.size = size
        self.shape = shape
        self.X0 = X0

    def info(self):
        """Prints information about the Circle mesh."""
        print(
            f"Circle mesh, shape: {self.shape}, X0: {self.X0}, radius: {self.size[0] / 2.0}"
        )


class Nonuniform(Mesh):
    """A mesh created by adaptively refining a parent mesh.

    This class refines a given `parent` mesh by adding new nodes inside cells
    that meet a certain `criterion`. It is useful for increasing resolution
    in specific areas of interest.
    """

    def __init__(
        self,
        parent: Mesh,
        values: Optional[torch.Tensor] = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], bool]] = None,
        weightening: Optional[Callable] = None,
        keep_parent: bool = False,
    ):

        if keep_parent:
            self.parent = parent
        if criterion is None:
            criterion = lambda mean, std: True
        if values is None:
            values = torch.ones(parent.N)

        cells = []
        new_positions = []
        new_weights = []
        self.criterion = criterion
        self.weightening = weightening
        n = parent.N
        for cell in parent.cells:

            nodes_ = cell
            vals_ = values[nodes_]
            new_pos, new_w = self.mean_strategy(
                parent.pos3d[nodes_], parent.weights[nodes_], vals_
            )
            if new_pos is not None:
                new_positions.append(new_pos)
                new_weights.append(new_w)
                cells.extend(self.new_cells(n, nodes_))
                n += 1
            else:
                cells.append(cell)

        num_new_points = n - parent.N
        generation = parent.generation + [parent.gen + 1] * num_new_points
        traced = torch.cat(
            (parent.traced, torch.zeros(num_new_points, dtype=torch.bool))
        )

        if num_new_points > 0:
            new_positions = torch.stack(new_positions)
            pos3d = torch.cat((parent.pos3d, new_positions), dim=0)

            new_weights = torch.stack(new_weights)
            weights = torch.cat((parent.weights, new_weights), dim=0)
        else:
            pos3d = parent.pos3d
            weights = parent.weights

        weights = weights / weights.sum()

        super().__init__(parent.anchor, pos3d, cells, weights, traced, generation)
        self.gen = parent.gen + 1
        self.uniform = False

    def info(self):
        """Prints information about the Nonuniform mesh."""
        print(f"Nonuniform mesh, nodes: {self.N}, gen: {self.gen}")

    def new_cells(self, new_node: int, old_nodes: List[int]) -> List[List[int]]:
        """
        Creates new cells by connecting the old cell's nodes to the new center node.
        This performs a fan triangulation of the original cell.
        """
        cells = []
        n_nodes = len(old_nodes)

        for i in range(n_nodes - 1):
            cells.append([old_nodes[i], old_nodes[i + 1], new_node])
        cells.append([old_nodes[-1], old_nodes[0], new_node])

        return cells

    def mean_strategy(
        self, positions: torch.Tensor, weights: torch.Tensor, values: torch.Tensor
    ):
        """
        A strategy for upsampling cells of the mesh.

        In this strategy, a criterion on the mean and std of values is
        used to make a decision on upsampling.

        If the condition succeeds, a new point will be the mean of the cell points
        and will be initialized with their mean weight.
        """
        mean = values.mean()
        std = values.std()
        if self.criterion(mean, std):
            return positions.mean(0), weights.mean(0)
        return None, None

    def weightened_strategy(self, values, positions, weights):
        """A placeholder for a weighted upsampling strategy."""
        raise NotImplementedError
