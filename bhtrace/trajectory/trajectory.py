from __future__ import annotations
from typing import List, Tuple, Optional, Union, Type
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import torch

from bhtrace.geometry.spacetime._base import Spacetime
from bhtrace.geometry.particle import Particle
from bhtrace.geometry.transformation import relation_dict
from bhtrace.graphics import Plot2D, PlotValue, Plot3D

if TYPE_CHECKING:
    from bhtrace.tracing._base import Tracer
    import matplotlib.axes as mpl_axes
    import matplotlib.figure as mpl_figure

@dataclass
class Trajectory:
    """A universal class for storing and operating with particle trajectory data.

    Attributes
    ----------
    X : torch.Tensor
        Trajectory coordinates, shape (ntraj, nsteps, 4).
    P : torch.Tensor
        Trajectory 4-momenta, shape (ntraj, nsteps, 4).
    affine_t : torch.Tensor
        Affine parameter values along trajectories, shape (ntraj, nsteps).
    particle : Particle
        The particle object that was traced.
    tracer : "Tracer"
        The tracer object that was used to generate the trajectory.
    coordinates : str, optional
        Name of the coordinate system for `X` and `P`. Defaults to the
        spacetime's default coordinate system.
    last_step : int, optional
        The last step index before all trajectories were stopped. Defaults to None.
    spacetime : Spacetime
        Spacetime in which the trajectory was traced.
    __coords__ : str
        The coordinate system used during tracing.
    __XP_reprs__ : dict
        Cache for different coordinate representations of `X` and `P`.
    ntraj : int
        Number of trajectories.
    nsteps : int
        Number of steps in each trajectory.
    """
    X: torch.Tensor
    P: torch.Tensor
    affine_t: torch.Tensor
    particle: Particle
    tracer: "Tracer"
    coordinates: Optional[str] = None
    last_step: Optional[int] = None

    # Attributes set in __post_init__
    particle_state: dict = field(init=False)
    spacetime: Spacetime = field(init=False)
    spacetime_state: dict = field(init=False)
    __coords__: str = field(init=False)
    tracer_state: dict = field(init=False)
    __XP_reprs__: dict = field(init=False, repr=False)
    ntraj: int = field(init=False)
    nsteps: int = field(init=False)

    def __post_init__(self) -> None:
        self.spacetime = self.particle.spacetime

        if self.coordinates is None:
            self.__coords__ = self.spacetime._coords
        else:
            self.__coords__ = self.coordinates

        self.X = self.X.detach().cpu()
        self.P = self.P.detach().cpu()
        self.affine_t = self.affine_t.detach().cpu()
        
        self.__XP_reprs__ = {}

        self.ntraj = self.X.shape[0]
        self.nsteps = self.X.shape[1]

    def __getitem__(self, key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if key == self.__coords__:
            return self.X, self.P
        elif key not in self.__XP_reprs__:
            try:
                transformation = relation_dict[self.__coords__][key]()
                X_new, P_new = transformation.tensor(self.X, self.P, valence=[False])
                self.__XP_reprs__[key] = (X_new, P_new)
                return X_new, P_new
            except KeyError:
                raise KeyError(f"No transformation is available between {self.__coords__} and {key}.")
        else:
            return self.__XP_reprs__[key]
    
    def __repr__(self) -> str:
        return f"Trajectory with {self.ntraj} particles and {self.nsteps} time slices."
    
    def __len__(self) -> int:
        return self.nsteps
    
    def join(self, trajectories: List[Trajectory], fill_reprs: bool = True) -> Trajectory:
        """Joins a list of trajectories to the current one.

        All trajectories must have the same number of steps, same coordinate
        system, and originate from the same particle and spacetime
        configuration.

        Parameters
        ----------
        trajectories : List[Trajectory]
            A list of Trajectory objects to join.
        fill_reprs : bool, optional
            If True, fills all coordinate representations present in any of the
            trajectories. If False, only keeps representations present in all
            of them. Defaults to True.

        Returns
        -------
        Trajectory
            The modified Trajectory object with joined data.
        """
        if not trajectories:
            return self

        # Compatibility checks
        for t in trajectories:
            if t.nsteps != self.nsteps:
                raise ValueError("Cannot join trajectories with different number of steps.")
            if t.__coords__ != self.__coords__:
                raise ValueError("Cannot join trajectories with different coordinate systems.")
            # if t.particle_state != self.particle_state:
            #     raise ValueError("Cannot join trajectories with different particle states.")
            # if t.spacetime_state != self.spacetime_state:
            #     raise ValueError("Cannot join trajectories with different spacetime states.")

        all_X = [self.X] + [t.X for t in trajectories]
        all_P = [self.P] + [t.P for t in trajectories]
        all_affine_t = [self.affine_t] + [t.affine_t for t in trajectories]

        all_last_step = [t.last_step for t in trajectories if t.last_step is not None]
        if self.last_step is not None:
            all_last_step.append(self.last_step)
        
        if len(all_last_step) > 0:
            self.last_step = max(all_last_step)

        self.X = torch.cat(all_X, dim=0)
        self.P = torch.cat(all_P, dim=0)
        self.affine_t = torch.cat(all_affine_t, dim=0)

        # Clear cached representations as they are now invalid.
        all_keys = set(self.__XP_reprs__.keys())
        for t in trajectories:
            _keys_ = set(t.__XP_reprs__.keys())
            if fill_reprs:
               all_keys.update(_keys_)
            else:
               all_keys.intersection_update(_keys_)

        for key in all_keys:
            new_X, new_P = self.__getitem__(key)
            reprs = [t.__getitem__(key) for t in trajectories]
            X_ = [new_X] + [_r_[0] for _r_ in reprs]
            P_ = [new_P] + [_r_[1] for _r_ in reprs]

            new_X = torch.cat(X_)
            new_P = torch.cat(P_)

            self.__XP_reprs__[key] = new_X, new_P

        self.ntraj += sum([t.ntraj for t in trajectories])
        return self

    def to(self, device: Union[torch.device, str]) -> Trajectory:
        """Moves the trajectory data to the specified device.

        Parameters
        ----------
        device : torch.device or str
            The device to move the tensors to.

        Returns
        -------
        Trajectory
            The trajectory object with data on the specified device.
        """
        self.X = self.X.to(device)
        self.P = self.P.to(device)
        self.affine_t = self.affine_t.to(device)
        return self
    
    @classmethod
    def from_dict(cls: Type[Trajectory], data: dict) -> Trajectory:
        """Creates a Trajectory object from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing trajectory data, typically loaded from a file.

        Returns
        -------
        Trajectory
            A new Trajectory object.
        """
        from bhtrace.tracing import MockTracer
        from bhtrace.geometry import Particle

        particle = Particle.from_dict(data['particle_state'])
        spacetime = particle.spacetime
        # TODO: Save tracer state
        tracer = MockTracer(particle, spacetime)
        
        affine_t = data.get('affine_t')
        if affine_t is None:
            affine_t = data.get('l')  # For backward compatibility
        if affine_t is None:
            # For backward compatibility with very old trajectory files
            affine_t = torch.zeros(data['X'].shape[0], data['X'].shape[1], dtype=data['X'].dtype, device=data['X'].device)

        traj = cls(data['X'], data['P'], affine_t, particle, tracer, data['coord_original'])
        if '__XP_reprs__' in data:
            traj.__XP_reprs__ = data['__XP_reprs__']
        
        if 'lens' in data:
            traj.lens = data['lens']

        return traj
    
    def save(self, filename: str, save_reprs: bool = True) -> None:
        """Saves the trajectory data to a file.

        Parameters
        ----------
        filename : str
            The path to the file where the trajectory data will be saved.
        save_reprs : bool, optional
            If True, saves all cached coordinate representations. Defaults to True.
        """
        data = {
            'X': self.X,
            'P': self.P,
            'affine_t': self.affine_t,
            'coord_original': self.__coords__,
            'particle_state': self.particle.state(),
            'spacetime_state': self.spacetime.state()
        }
        if save_reprs:
            data['__XP_reprs__'] = self.__XP_reprs__
        if hasattr(self, 'lens'):
            data['lens'] = self.lens

        torch.save(data, filename)
        print(f'File saved at {filename}')
    
    @staticmethod
    def load(filename: str) -> Trajectory:
        """Loads trajectory data from a file.

        Parameters
        ----------
        filename : str
            The path to the file to load.

        Returns
        -------
        Trajectory
            A Trajectory object loaded from the file.
        """
        data = torch.load(filename)
        return Trajectory.from_dict(data)

    def plot2d(self, ax: Optional["mpl_axes.Axes"] = None, figsize: Tuple[int, int] = (10, 10), **kwargs) -> "mpl_figure.Figure":
        """Plots 2D trajectories on a matplotlib axis.

        If no axis is provided, a new figure and axis are created.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If None, a new figure and axis are created.
            Defaults to None.
        figsize : tuple, optional
            The size of the figure to create if `ax` is None.
            Defaults to (10, 10).
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        return Plot2D.plot_2d(self, ax=ax, figsize=figsize, **kwargs)

    @classmethod
    def plot2d_(cls: Type[Trajectory], trajectories: List[Trajectory], figsize: Tuple[int, int] = (10, 10), **kwargs) -> "mpl_figure.Figure":
        """Plots multiple trajectories on a mosaic of subplots.

        Parameters
        ----------
        trajectories : List[Trajectory]
            A list of Trajectory objects to plot.
        figsize : tuple, optional
            The size of the figure to create. Defaults to (10, 10).
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plots.
        """
        return Plot2D.plot_2d_mosaic(trajectories, figsize=figsize, **kwargs)
    
    def plot_metrics(self) -> "mpl_figure.Figure":
        """Plots integrator-specific metrics along the trajectory.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        return PlotValue.plot_metrics(self)

    def plot_energy_deviation(self, ax: Optional["mpl_axes.Axes"] = None) -> "mpl_figure.Figure":
        """Plots energy deviation vs time along the trajectory for each ray.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        return PlotValue.plot_energy_deviation(self, ax=ax)
    
    def plot_energy_deviation_histogram(self, ax: Optional["mpl_axes.Axes"] = None) -> "mpl_figure.Figure":
        """Plots a histogram of energy deviation along the trajectory.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        return PlotValue.plot_energy_deviation_histogram(self, ax=ax)
    
    def plot3d(self, fig: Optional["mpl_figure.Figure"] = None, ax: Optional["mpl_axes.Axes"] = None) -> "mpl_figure.Figure":
        """Plots 3D trajectories.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. If None, a new figure is created.
        ax : matplotlib.axes.Axes, optional
            The 3D subplot to plot on. If None, a new subplot is created.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        
        return Plot3D.lines(
            points = self['Cartesian'][0][..., 1:],
            fig = fig,
            ax = ax,
        )
   
    def report(self) -> dict:
        """Generates a report with several plots of the trajectory.

        The report includes a 2D plot, conservation plots, and metric plots.

        Returns
        -------
        dict
            A dictionary of figure objects, with keys '2d', 'conservation',
            and 'metrics'.
        """

        figs = {}
        figs['2d'] = self.plot2d()
        figs['conservation'] = self.plot_energy_deviation()
        figs['metrics'] = self.plot_metrics()

        return figs



