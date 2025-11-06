'''
This module provides classes for generating and managing coordinate nets.

These nets are primarily used to define the initial conditions for tracing
trajectories, for instance, by defining a grid of starting points for photons
on an observer's screen.

The module includes a base `Net` class and implementations for various shapes
such as `Linear`, `Rectangle`, and adaptively refined `Nonuniform` nets.
'''
from abc import ABC, abstractmethod
from typing import Tuple, List, Callable, Optional

import torch
from bhtrace.graphics import Plot3D
from bhtrace.utils.debug import debug


class Mesh(ABC):
    """Abstract base class for all coordinate mesh implementations.

    This class defines the common interface and attributes for all nets,
    such as node positions, weights, and cell structures.

    Attributes:
        pos3d (torch.Tensor): A tensor of shape (N, 3) holding the Cartesian coordinates of the N nodes.
        cells (List[List[int]]): A list of cells, where each cell is a list of node indices.
        gen (int): The number of upsampling generations this net has undergone.
        uniform (bool): A flag indicating if the net has uniform spacing.
        N (int): The total number of nodes in the net.
        anchor (torch.Tensor): The anchor point of the net, used for positioning or rotation.
        weights (torch.Tensor): A tensor of shape (N,) representing the area weight of each node.
        traced (torch.Tensor): A boolean tensor of shape (N,) used as a mask to indicate which nodes have been traced.
        generation (List[int]): A list of length N indicating the upsampling generation of each node.
    """

    pos3d: torch.Tensor = None
    gen: int = 0
    '''Number of refinement steps done'''

    '''Weights of each vertex, must sum up to 1'''
    uniform: bool = True

    cells: List[List[int]] = None
    active_cells: list[int]
    traced: list[int]

    def __init__(
            self,
            size: torch.Tensor = torch.Tensor([20, 20]),
            discretization:  Tuple[int, int] = (16, 16),
            anchor: torch.Tensor = torch.zeros(3), 
            vertxs: torch.Tensor = None,
            cells : torch.Tensor = None,
            weights: Optional[torch.Tensor] = None, 
            traced: Optional[torch.Tensor] = None, 
            generation: Optional[List[int]] = None,
        ):
        self.size = size
        self.D = discretization
        self.N = self.D[0]*self.D[1]
        self.anchor: torch.Tensor = anchor
        self.pos3d = vertxs
        self.cells = cells
        self.weights: torch.Tensor = weights if weights is not None else torch.ones(self.N)
        self.traced: torch.Tensor = traced if traced is not None else torch.zeros(self.N, dtype=torch.bool)
        self.generation: List[int] = generation if generation is not None else [0] * self.N

    @classmethod
    def generate_initial(self, ):

        return NotImplementedError

    @abstractmethod
    def info(self):
        """Prints information about the net to the console."""
        pass

    def to(self, device=None, dtype=None):
        """Moves and/or casts the tensors of the net.

        This method iterates over all tensor attributes of the net and moves them
        to the specified device. For floating-point tensors, it also casts them
        to the specified dtype.

        Args:
            device (str, optional): The device to move the tensors to.
            dtype (torch.dtype, optional): The data type to cast floating-point tensors to.

        Returns:
            Net: The modified net object.
        """
        for attr_name in dir(self):
            if attr_name.startswith('__'):
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
        """Visualizes the net as a 3D point cloud.

        Args:
            direction: (Not yet implemented).
            fig: An existing matplotlib figure.
            ax: An existing matplotlib 3D axes.

        Returns:
            A tuple containing the figure and axes.
        """
        # Integrate to point cloud or implement new plot?
        # x, y = self.vertices[:, 0].numpy(), self.vertices[:, 1].numpy()
        # triangles = self.faces.numpy()
        
        # ax.set_aspect('equal')
        # ax.set_title(title)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")

        # if show_values:
        #     avg_face_values = self.vertex_values[self.faces].mean(dim=1).numpy()
        #     tripcolor = ax.tripcolor(x, y, triangles, facecolors=avg_face_values, cmap='viridis')
        #     fig.colorbar(tripcolor, ax=ax, label='Vertex Value')
        
        # ax.triplot(x, y, triangles, 'k-', lw=0.5)
        return Plot3D.point_cloud(self.pos3d, self.generation, fig=fig, ax=ax)

    def refine(
        self,
        values: Optional[torch.Tensor] = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], bool]] = None,
        weightening: Optional[Callable] = None,
        nsteps: int = 1,
        interpolation: Optional[Callable] = None
        ):
        """Creates a new, adaptively refined mesh from the current mesh.

        Args:
            values (torch.Tensor, optional): A tensor of values at each vertex, used by the criterion to decide where to refine. \
            If not provided, mesh will be refined uniformly;
            criterion (callable, optional): A function that takes the mean and std of values in a cell and returns True to refine. \
            If not provided, condition std > eps will be used to determine which cells to refine;
            weightening (callable, optional): A function to determine the weights of new nodes. \
            Mean by default;
            nsteps (int): The number of refinement steps to perform;
            interpolation: function to interpolate values of new vertices in case of multistep interpolation.

        Returns:
            The new, refined net.
        """

        active_ids = torch.where(self.active_cells)[0]
        if len(active_ids) == 0:
            print("No active triangles to refine.")
            return None

        triangles_to_refine_indices = []
        for tri_idx in active_ids:
            face_vertex_indices = self.faces[tri_idx]
            face_values = self.vertex_values[face_vertex_indices]
            
            if torch.std(face_values) > eps:
                triangles_to_refine_indices.append(tri_idx)
            else:
                self.face_active_status[tri_idx] = False

        if not triangles_to_refine_indices:
            print("No triangles met the refinement condition.")
            return

        new_vertices = []
        new_vertex_weights = []
        new_faces_list = self.faces.tolist()
        new_face_active_status_list = self.face_active_status.tolist()

        edge_to_new_vertex_map = {}
        faces_to_remove_indices = set()

        for tri_index in triangles_to_refine_indices:
            faces_to_remove_indices.add(tri_index.item())
            face = self.faces[tri_idx]
            v_indices = face
            v_coords = self.vertices[v_indices]

            edge_lengths_sq = [torch.sum((v_coords[1] - v_coords[0])**2), torch.sum((v_coords[2] - v_coords[1])**2), torch.sum((v_coords[0] - v_coords[2])**2)]
            longest_edge_local_idx = edge_lengths_sq.index(max(edge_lengths_sq))
            
            v_idx1 = v_indices[longest_edge_local_idx]
            v_idx2 = v_indices[(longest_edge_local_idx + 1) % 3]
            edge = tuple(sorted((v_idx1.item(), v_idx2.item())))

            if edge in edge_to_new_vertex_map:
                new_vertex_global_idx = edge_to_new_vertex_map[edge]
            else:
                new_coord = (self.vertices[v_idx1] + self.vertices[v_idx2]) / 2.0
                new_weight = (self.vertex_weights[v_idx1] + self.vertex_weights[v_idx2]) / 2.0
                
                new_vertices.append(new_coord)
                new_vertex_weights.append(new_weight)
                
                new_vertex_global_idx = len(self.vertices) + len(new_vertices) - 1
                edge_to_new_vertex_map[edge] = new_vertex_global_idx

            v_idx3 = v_indices[(longest_edge_local_idx + 2) % 3]
            
            new_faces_list.append([v_idx1.item(), new_vertex_global_idx, v_idx3.item()])
            new_faces_list.append([v_idx2.item(), new_vertex_global_idx, v_idx3.item()])
            new_face_active_status_list.extend([True, True])

        if new_vertices:
            self.vertices = torch.cat([self.vertices, torch.stack(new_vertices)], dim=0)
            # Recalculate all vertex values from the function
            self.vertex_values = calculate_vertex_values(self.vertices)
            self.vertex_weights = torch.cat([self.vertex_weights, torch.tensor(new_vertex_weights)], dim=0)

        final_faces = [face for i, face in enumerate(new_faces_list) if i not in faces_to_remove_indices]
        final_active_status = [status for i, status in enumerate(new_face_active_status_list) if i not in faces_to_remove_indices]
        
        self.faces = torch.tensor(final_faces, dtype=torch.long)
        self.face_active_status = torch.tensor(final_active_status, dtype=torch.bool)


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
        super().__init__(npoints, anchor)
        self.shape = (npoints,)
        self.size = size
        self.X0 = X0

        self.pos3d = X0 + torch.linspace(-0.5, 0.5, npoints).unsqueeze(-1) * size.unsqueeze(0)

        self.cells = [[i, i+1] for i in range(npoints-1)]

    def info(self):
        """Prints information about the Linear net."""
        print(f"Linear net, shape: {self.shape}, X0: {self.X0}")

class Rectangle(Mesh):
    """A 2D, rectangular grid of nodes.

    Args:
        size (torch.Tensor): A 2-element tensor defining the width and height of the rectangle.
        shape (Tuple[int, int]): The number of nodes along the y and z dimensions.
        X0 (torch.Tensor): The 3D coordinate of the rectangle's center.
        anchor (torch.Tensor): The anchor point of the net.
    """

    def __init__(
            self,
            size: torch.Tensor,
            shape: Tuple[int, int] = (5, 5), 
            X0: torch.Tensor = torch.zeros(3),
            anchor: torch.Tensor = torch.zeros(3),
            ):
        super().__init__(shape[0]*shape[1], anchor)
        self.size = size
        self.shape = shape
        self.X0 = X0

        y_ = torch.linspace(-0.5, 0.5, shape[0])*size[0]
        z_ = torch.linspace(-0.5, 0.5, shape[1])*size[1]
        y_grid, z_grid = torch.meshgrid(y_, z_, indexing='ij')
        x_grid = torch.zeros_like(y_grid)
        
        self.pos3d = X0 + torch.stack((x_grid, y_grid, z_grid), dim=-1)
        self.pos3d = self.pos3d.flatten(0,-2)
        # Writing node indexes in clockwise order:
        self.cells = [[k*shape[1]+i, k*shape[1]+i+1, (k+1)*shape[1]+i+1, (k+1)*shape[1]+i]\
                      for i in range(shape[1]-1) for k in range(shape[0]-1)]

    def info(self):
        """Prints information about the Rectangle net."""
        print(f"Rectangle net, shape: {self.shape}, X0: {self.X0}")

import math

class Hex(Mesh):
    """A 2D, hexagonal grid of nodes.

    This implementation uses axial coordinates to generate a "pointy-top" hexagonal
    grid arranged in a parallelogram shape.

    Args:
        size (torch.Tensor): A 2-element tensor defining the approximate width and height of the grid.
        shape (Tuple[int, int]): The number of columns (q) and rows (r) in the axial coordinate system.
        X0 (torch.Tensor): The 3D coordinate of the grid's center.
        anchor (torch.Tensor): The anchor point of the net.
    """

    def __init__(
            self,
            size: torch.Tensor,
            shape: Tuple[int, int] = (5, 5), 
            X0: torch.Tensor = torch.zeros(3),
            anchor: torch.Tensor = torch.zeros(3),
            ):
        n_q, n_r = shape
        N = n_q * n_r
        super().__init__(N, anchor)

        self.size = size
        self.shape = shape
        self.X0 = X0

        # Derive hex_size from overall width
        # A grid of n_q pointy-top hexes has a width of hex_size * (1.5 * (n_q - 1) + 2)
        # We approximate this to 1.5 * n_q for simplicity.
        hex_size = self.size[0] / (1.5 * n_q)

        # Generate points
        positions = []
        for q in range(n_q):
            for r in range(n_r):
                # Convert axial (q, r) to cartesian (y, z)
                y = hex_size * 1.5 * q
                z = hex_size * math.sqrt(3) * (r + q / 2.0)
                positions.append([0, y, z])
        
        self.pos3d = torch.tensor(positions, dtype=torch.float32)
        
        # Recenter the grid
        if N > 0:
            mean_pos = self.pos3d.mean(dim=0)
            self.pos3d -= mean_pos

        self.pos3d += X0

        # Define cells as triangles
        self.cells = []
        for q in range(n_q - 1):
            for r in range(n_r - 1):
                p1 = q * n_r + r         # index of (q, r)
                p2 = (q + 1) * n_r + r   # index of (q+1, r)
                p3 = q * n_r + (r + 1)   # index of (q, r+1)
                p4 = (q + 1) * n_r + (r + 1) # index of (q+1, r+1)
                
                # Create two triangles from the rhombus
                self.cells.append([p1, p2, p3])
                self.cells.append([p2, p4, p3])

    def info(self):
        """Prints information about the Hex net."""
        print(f"Hex net, shape: {self.shape}, X0: {self.X0}")
    
class Circle(Mesh):
    """A 2D, polar grid of nodes.

    The grid is composed of a central point, surrounded by concentric rings.

    Args:
        size (torch.Tensor): A 1-element tensor defining the diameter of the circle.
        shape (Tuple[int, int]): The number of rings and the number of wedges (radial divisions).
        X0 (torch.Tensor): The 3D coordinate of the circle's center.
        anchor (torch.Tensor): The anchor point of the net.
    """

    def __init__(
            self,
            size: torch.Tensor,
            shape: Tuple[int, int] = (5, 5), 
            X0: torch.Tensor = torch.zeros(3),
            anchor: torch.Tensor = torch.zeros(3),
            ):
        n_rings, n_wedges = shape
        N = 1 + n_rings * n_wedges
        super().__init__(N, anchor)

        self.size = size
        self.shape = shape
        self.X0 = X0
        
        ph = [torch.linspace(0.0, 2.0, 4*(n+1)+1)[1:] for n in range(rng[0]-1)]
        ph = torch.cat(ph)*torch.pi
        r = [torch.ones(4*(n+1)+1)[1:]*(n+1)/rng[0] for n in range(rng[0]-1)]
        r = torch.cat(r)
        yy = r*torch.sin(ph)
        zz = r*torch.cos(ph)

        radius = self.size[0] / 2.0
        
        # Create central point
        center_point = torch.tensor([[0.0, 0.0, 0.0]])

        # Create points on rings
        if n_rings > 0:
            radii = torch.linspace(radius / n_rings, radius, n_rings)
            angles = torch.linspace(0, 2 * torch.pi, n_wedges + 1)[:-1]
            r_grid, a_grid = torch.meshgrid(radii, angles, indexing='ij')
            
            y = r_grid * torch.cos(a_grid)
            z = r_grid * torch.sin(a_grid)
            x = torch.zeros_like(y)
            
            ring_points = torch.stack((x, y, z), dim=-1).flatten(0, -2)
            self.pos3d = X0 + torch.cat((center_point, ring_points), dim=0)
        else:
            self.pos3d = X0 + center_point

        # Define cells
        self.cells = []
        if n_rings > 0:
            # Inner triangular cells
            for j in range(n_wedges):
                p1 = 1 + j
                p2 = 1 + (j + 1) % n_wedges
                self.cells.append([0, p1, p2])

            # Outer quadrilateral cells
            for i in range(n_rings - 1):
                for j in range(n_wedges):
                    p1 = 1 + i * n_wedges + j
                    p2 = 1 + i * n_wedges + (j + 1) % n_wedges
                    p3 = 1 + (i + 1) * n_wedges + (j + 1) % n_wedges
                    p4 = 1 + (i + 1) * n_wedges + j
                    self.cells.append([p1, p2, p3, p4])

    def info(self):
        """Prints information about the Circle net."""
        print(f"Circle net, shape: {self.shape}, X0: {self.X0}, radius: {self.size[0]/2.0}")

class Nonuniform(Mesh):
    """A net created by adaptively refining a parent net.

    This class refines a given `parent` net by adding new nodes inside cells
    that meet a certain `criterion`. It is useful for increasing resolution
    in specific areas of interest.
    """

    def __init__(self,
                 parent: Net,
                 values: Optional[torch.Tensor] = None,
                 criterion: Optional[Callable[[torch.Tensor, torch.Tensor], bool]] = None,
                 weightening: Optional[Callable] = None,
                 keep_parent: bool = False
                 ):
        
        if keep_parent:
            self.parent = parent
        if criterion is None:
            criterion = lambda mean, std: True
        if values is None:
            values = torch.ones(parent.N)
        
        self.cells = []
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
                self.cells.extend(self.new_cells(n, nodes_))
                n +=1
            else:
                self.cells.append(cell)

        num_new_points = n - parent.N
        generation = parent.generation + [parent.gen + 1] * num_new_points
        traced = torch.cat((parent.traced, torch.zeros(num_new_points, dtype=torch.bool)))

        super().__init__(n, parent.anchor, generation=generation, traced=traced)
        self.gen = parent.gen + 1
        self.uniform = False

        if num_new_points > 0:
            new_positions = torch.stack(new_positions)
            self.pos3d = torch.cat((parent.pos3d, new_positions), dim=0)

            new_weights = torch.stack(new_weights)
            self.weights = torch.cat((parent.weights, new_weights), dim=0)
        else:
            self.pos3d = parent.pos3d
            self.weights = parent.weights
        
        self.weights = self.weights/self.weights.sum()

    def info(self):
        """Prints information about the Nonuniform net."""
        print(f"Nonuniform net, nodes: {self.N}, gen: {self.gen}")
    
    def new_cells(self, new_node: int, old_nodes: List[int]) -> List[List[int]]:
        '''
        Creates new cells by connecting the old cell's nodes to the new center node.
        This performs a fan triangulation of the original cell.
        '''
        cells = []
        n_nodes = len(old_nodes)

        for i in range(n_nodes-1):
            cells.append([old_nodes[i], old_nodes[i+1], new_node])
        cells.append([old_nodes[-1], old_nodes[0], new_node])

        return cells

    def mean_strategy(self, positions: torch.Tensor, weights: torch.Tensor, values: torch.Tensor):
        """
        A strategy for upsampling cells of network.

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
    