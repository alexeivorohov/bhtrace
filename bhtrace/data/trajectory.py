from __future__ import annotations
from typing import List, Tuple, Optional, Union, Type, Literal
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from functools import cached_property, cache, lru_cache

import torch
import numpy as np

from bhtrace.geometry.spacetime._base import Spacetime
from bhtrace.geometry.particle import Particle
from bhtrace.geometry.transformation import relation_dict
import bhtrace.graphics as bhg

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from bhtrace.tracing._base import Tracer
    from bhtrace.data.grrtdata import GRRTData

# TODO: add get_scalar interface which will map keys to scalar features

@dataclass
class Trajectory:
    """A universal class for storing and operating with particle trajectory data.

    Caution: after calculation, `X`, `P` and `V` should be treated as immutable.

    Attributes
    ----------
    X : torch.Tensor
        Trajectory coordinates, shape (ntraj, nsteps, 4).
    P : torch.Tensor
        Trajectory 4-momenta, shape (ntraj, nsteps, 4).
    V : torch.Tensor
        Trajectory 4-velocities, shape (ntraj, nsteps, 4).

    affine_t : torch.Tensor
        Affine parameter values along trajectories, shape (nsteps,).
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
    solution_coordinates : str
        The coordinate system used during tracing.
    __XP_reprs__ : dict
        Cache for different coordinate representations of `X` and `P`.
    ntraj : int
        Number of trajectories.
    nsteps : int
        Number of steps in each trajectory.
    """

    _X: torch.Tensor = field(init=True, repr=False)
    _P: torch.Tensor = field(init=True, repr=False)
    affine_t: torch.Tensor = field(init=True, repr=False)
    particle: Particle = field(init=True, repr=False)
    tracer: "Tracer" = field(init=True, repr=False)
    _genuine_steps: torch.Tensor[bool] = field(init=True, repr=False)


    last_step: Optional[int] = None
    solution_coordinates: str = field(init=False, repr=True)
    ntraj: int = field(init=False, repr=True)
    nsteps: int = field(init=False, repr=True)
    particle_state: dict = field(init=False, repr=False)
    spacetime: Spacetime = field(init=False, repr=False)
    spacetime_state: dict = field(init=False, repr=False)
    tracer_state: dict = field(init=False, repr=False)
    __XP_reprs__: dict = field(init=False, repr=False)


    @cached_property
    def energy(self, frame: torch.Tensor = None) -> torch.Tensor:
        """Particle energy along the trajectory"""
        if frame is None:
            frame = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._X.device)
        return self.particle.energy(self._X, self._P, frame)

    @cached_property
    def metric(self) -> torch.Tensor:
        """Metric components along the trajectory"""
        return self.spacetime.g(self._X)

    @cached_property
    def mu_violation(self) -> torch.Tensor:
        """Violation of the mass-shell condition along the trajectory

        Currently includes significiant contribution from stopped trajectories.
        """

        g = self.spacetime.ginv(self._X)

        dlta_mu = (
            torch.einsum("...a,...b,...ab->...", self._P, self._P, g)
            + self.particle.mu
        )

        return dlta_mu


    def __post_init__(self) -> None:
        self.spacetime = self.particle.spacetime
        self.solution_coordinates = self.spacetime._coords
        
        self._X = self._X.detach().cpu()
        self._P = self._P.detach().cpu()
        self.affine_t = self.affine_t.detach().cpu()
        self._genuine_steps = self._genuine_steps.detach().cpu()

        self.ntraj = self._X.shape[0]
        self.nsteps = self._X.shape[1]
        self.__XP_reprs__ = {}

    @property
    def X(self) -> torch.Tensor:
        return self._X.clone()
    
    @property
    def P(self) -> torch.Tensor:
        return self._P.clone()

    def __getitem__(self, key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns X and P representation in coordinate system, given by `key`

        A representation will be automaticaly cached for further use.

        This cache can be saved with the trajectory.
        """
        if key == self.solution_coordinates:
            return self.X, self.P
        
        if key not in self.__XP_reprs__:
            transformation = relation_dict[self.solution_coordinates][key]()
            print(f'Called transform from {self.solution_coordinates} to {key} coordinates')
            X_new, P_new = transformation.tensor(self.X, self.P, valence=[False])
            self.__XP_reprs__[key] = (X_new, P_new)
            return X_new, P_new
            
        return self.__XP_reprs__[key]
      
    def __repr__(self) -> str:
        """String representation of the Trajectory instance"""
        return f"\
            Trajectory(ntraj={self.ntraj}, nsteps={self.nsteps}, solution_coordinates={self.solution_coordinates})\
        \
            "
    
    def __len__(self) -> int:
        """Number of steps in this trajectory"""
        return self.nsteps

    # --- Data manipulation ---

    # TODO: add more keys or provide another solution
    def get_scalar(self, key: str) -> torch.Tensor:
        """A helper method to map scalar quantities for plotting and analysis."""
            
        match key:
            case "energy":
                return self.energy
            case "mu_violation":
                return self.mu_violation
            case "affine_time":
                q = self.affine_t
            case "proper_time":
                q = None  # TODO: Implement proper time calculation
            case "detg":
                q = self.metric.det()
            case "g00":
                q = self.metric[..., 0, 0] # Handle other cases?
            case _:
                raise KeyError(f"No cached scalar quantity with key '{key}'")
        return q

        
    def projection2d(
            self, projection: str | torch.Tensor = 'xy', coordinates: str = 'Cartesian'
    ) -> torch.Tensor:
        """Project Trajectory onto 2d surface
        
        Parameters
        ----------
        projection : str or torch.Tensor (default='xy')
            Projection key or projection matrix
        coordinates : str (default='xy)
            Which coordinate representation to use for projection

        Returns
        -------
        torch.Tensor of shape(ntraj, nsteps, 2)
            Projected coordinates
        """
        ...
        

    # TODO: Renew existing representations or add argument to control this behaviour
    def interpolate(self, expansion_factor: float = 2.0) -> Trajectory:
        """Interpolates the trajectory data to increase the number of steps.

        Parameters
        ----------
        expansion_factor : float, optional
            The factor by which to increase the number of steps. Defaults to 2.0.

        Returns
        -------
        Trajectory
            The modified Trajectory object with interpolated data.
        """
        new_nsteps = int(self.nsteps * expansion_factor)
        new_affine_t = torch.linspace(
            self.affine_t[:, 0].min(),
            self.affine_t[:, -1].max(),
            new_nsteps,
            device=self.affine_t.device,
        )

        new_X = torch.zeros((self.ntraj, new_nsteps, 4), device=self._X.device)
        new_P = torch.zeros((self.ntraj, new_nsteps, 4), device=self._P.device)

        for i in range(self.ntraj):
            for j in range(4):
                new_X[i, :, j] = torch.interp(
                    new_affine_t, self.affine_t[i], self._X[i, :, j]
                )
                new_P[i, :, j] = torch.interp(
                    new_affine_t, self.affine_t[i], self._P[i, :, j]
                )

        outp = self.__class__(
            X=new_X,
            P=new_P,
            affine_t=new_affine_t,
            particle=self.particle,
            tracer=self.tracer,
            coordinates=self.solution_coordinates,
            last_step=self.last_step,
        )

        return outp

    def to(self, device: Union[torch.device, str], dtype: str) -> Trajectory:
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
        self._X = self._X.to(device)
        self._P = self._P.to(device)
        self.affine_t = self.affine_t.to(device)
        return self

    # --- Save and load ---
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

        particle = Particle.from_dict(data["particle_state"])
        spacetime = particle.spacetime
        # TODO: Save tracer state
        tracer = MockTracer(particle, spacetime)

        affine_t = data.get("affine_t")
        if affine_t is None:
            affine_t = data.get("l")  # For backward compatibility
        if affine_t is None:
            # For backward compatibility with very old trajectory files
            affine_t = torch.zeros(
                data["X"].shape[0],
                data["X"].shape[1],
                dtype=data["X"].dtype,
                device=data["X"].device,
            )

        traj = cls(
            data["X"], data["P"], affine_t, particle, tracer, data["coord_original"]
        )
        if "__XP_reprs__" in data:
            traj.__XP_reprs__ = data["__XP_reprs__"]

        if "lens" in data:
            traj.lens = data["lens"]

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
            "X": self._X,
            "P": self._P,
            "affine_t": self.affine_t,
            "coord_original": self.solution_coordinates,
            "particle_state": self.particle.state(),
            "spacetime_state": self.spacetime.state(),
        }
        if save_reprs:
            data["__XP_reprs__"] = self.__XP_reprs__
        if hasattr(self, "lens"):
            data["lens"] = self.lens

        torch.save(data, filename)
        print(f"File saved at {filename}")

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

    # --- Plotting routines ---

    # TODO: Consider introducing projection argument here?
    # TODO: get horizon slice from spacetime object.
    def plot2d(
        self,
        color_by_value: str = None,
        scope: Tuple[float, float, float, float] = (-20, 20, -20, 20),
        cleaned: bool = False,
        label: Optional[str] = None,
        scale: str = 'linear',
        projection: str | np.ndarray = 'xy',
        fig: Optional[plt.Figure] = None,
        sm = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots 2D trajectories on a matplotlib axis.

        If no axis is provided, a new figure and axis are created.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. If None, a new figure is created.
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

        horizon = 2.0 # Temporary hardcoded horizon radius. 

        x, _ = self['Cartesian']
        coords_np = x.numpy()

        q = None
        if color_by_value:
            q = self.get_scalar(color_by_value).numpy()

        # if cleaned:
        #     q = q[self._genuine_steps]

        # xy = self.projection2d(projection=projection, coordinates='Cartesian')

        fig, ax = bhg.plot2d(
            coords_np[..., 1:3],
            q=q,
            q_label = color_by_value,
            horizon=horizon,
            label=label,
            borders=scope,
            q_scale=scale,
            fig=fig,
            ax=ax,
            **kwargs
        )

        return fig, ax

    def plot3d(
        self,
        color_by_value: str = None,
        scope: float = 20,
        cleaned: bool = False,
        label: Optional[str] = None,
        cmap: str = 'viridis',
        sm = None,
        fig: Optional[plt.Axes] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
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

        x, _ = self['Cartesian']
        coords_np = x.numpy()

        q = None
        if color_by_value:
            q = self.get_scalar(color_by_value)

        return bhg.plot3d(
            xyz=coords_np[..., 1:],
            q=q,
            cmap=cmap,
            sm=sm,
            fig=fig,
            ax=ax,
        )

    def histogram(
        self,
        quantity: str = 'mu_violation',
        bins: int | np.ndarray = 16,
        q_scale: str = 'linear',
        p_scale: str = 'linear',
        density: bool = True,
        cleaned: bool = True,
        label: str = None,
        info_text: str = None,
        backend: Literal['mpl', 'uniplot'] = 'mpl',
        replace_nan: float = None,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots distribution histogram for given quantities

        Parameters
        ----------

        """
        values = self.get_scalar(quantity)

        if cleaned:
            values = values[self._genuine_steps]

        if replace_nan is not None:
            values.nan_to_num(replace_nan)
        else:
            values = values[- values.isnan()]

        return bhg.hist(
            data = values.abs().numpy().flatten(),
            bins=bins,
            backend = backend,
            density = density,
            q_scale=q_scale,
            p_scale=p_scale,
            label = label or quantity,
            info_text = info_text,
            fig = fig,
            ax = ax,
            **kwargs,
        )
    
    def ridge(
        self,
        quantity: Literal['energy', 'mu_violation'] = 'mu_violation',
        bins: int | np.ndarray = 16,
        q_scale: str = 'linear',
        p_scale: str = 'linear',
        density: bool = True,
        cleaned: bool = True,
        label: str = None,
        info_text: str = None,
        backend: Literal['mpl', 'uniplot'] = 'mpl',
        replace_nan: float = None,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots a sequence of distribution histograms for given quantity along time or batch
        
        Parameters
        ----------
        quantity : str
            The quantity to plot. Must be a key for get_scalar.
        bins : int or array-like
            The bin specification for the histogram. Passed to numpy.histogram.
        q_scale : str
            The scale for the quantity axis. Passed to matplotlib. Defaults to 'linear'.
        p_scale : str
            The scale for the probability axis. Passed to matplotlib. Defaults to 'linear'.
        density : bool
            A flag to compute distribution density instead of count numbers
        cleaned : bool
            A flag to extract only data for valid trajectory steps (e.g. ingores all padding after stop condition)
        label: str
            Histogram label
        backend: 'mpl' or 'uniplot'
            Graphic backend to use. `mpl` is for matplotlib backend - well suited for storing and publishing plots. 
            `uniplot` is well suited for in-shell visualization. See `bhtrace.graphics.histogram` for details.
        replace_nan: float
            If not None, replaces NaN values in the quantity tensor with this value.
        fig: matplotlib.figure.Figure, optional
            The figure to plot on. If None, a new figure is created.
        ax: matplotlib.axes.Axes, optional 
            The axis to plot on. If None, a new figure and axis are created.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        pass
    
    

    def plot_solution(self, coords: str) -> plt.Figure:
        """Plots solution coordinates and impulses in given system

        Parameters
        ----------
        coords : str
        idxs : int | List[int], optional

        Returns
        -------
        plt.Figure
        """

        x, p = self[coords]
        


        pass