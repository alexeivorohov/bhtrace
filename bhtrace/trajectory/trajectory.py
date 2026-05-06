from __future__ import annotations
from typing import List, Tuple, Optional, Union, Type, Literal
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from functools import cached_property

import torch
import numpy as np

from bhtrace.geometry.spacetime._base import Spacetime
from bhtrace.geometry.particle import Particle
from bhtrace.geometry.transformation import relation_dict
import bhtrace.graphics as bhg

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from bhtrace.tracing._base import Tracer

# TODO: add _get_cached_scalar interface which will map keys to scalar features
# TODO: __coords__ and coordinates are duplicated properties

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
    _genuine_steps: torch.Tensor[bool] = field(init=True, repr=False) # BUG: Does not initialized properly

    # Attributes set in __post_init__
    coordinates: Optional[str] = None
    last_step: Optional[int] = None
    __coords__: str = field(init=False)
    ntraj: int = field(init=False)
    nsteps: int = field(init=False)
    particle_state: dict = field(init=False, repr=False)
    spacetime: Spacetime = field(init=False, repr=False)
    spacetime_state: dict = field(init=False, repr=False)
    tracer_state: dict = field(init=False, repr=False)
    __XP_reprs__: dict = field(init=False, repr=False)


    @cached_property
    def energy(self, frame: torch.Tensor = None) -> torch.Tensor:
        """Particle energy along the trajectory"""
        if frame is None:
            frame = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.X.device)
        return self.particle.energy(self.X, self.P, frame)

    @cached_property
    def metric(self) -> torch.Tensor:
        """Metric components along the trajectory"""
        return self.spacetime.g(self.X)

    @cached_property
    def mu_violation(self) -> torch.Tensor:
        """Violation of the mass-shell condition along the trajectory

        Currently includes significiant contribution from stopped trajectories.
        """

        g = self.spacetime.ginv(self.X)

        dlta_mu = (
            torch.einsum("...a,...b,...ab->...", self.P, self.P, g)
            + self.particle.mu
        )

        return dlta_mu

    def __post_init__(self) -> None:
        self.spacetime = self.particle.spacetime

        if self.coordinates is None:
            self.__coords__ = self.spacetime._coords
        else:
            self.__coords__ = self.coordinates

        self.X = self.X.detach().cpu()
        self.P = self.P.detach().cpu()
        self.affine_t = self.affine_t.detach().cpu()
        self._genuine_steps = self._genuine_steps.detach().cpu()

        self.__XP_reprs__ = {}

        self.ntraj = self.X.shape[0]
        self.nsteps = self.X.shape[1]


    def __getitem__(self, key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns X and P representation in coordinate system, given by `key`

        A representation will be automaticaly cached for further use.

        This cache can be saved with the trajectory.
        """
        if key == self.__coords__:
            return self.X, self.P
        elif key not in self.__XP_reprs__:
            try:
                transformation = relation_dict[self.__coords__][key]()
                X_new, P_new = transformation.tensor(self.X, self.P, valence=[False])
                self.__XP_reprs__[key] = (X_new, P_new)
                return X_new, P_new
            except KeyError:
                raise KeyError(
                    f"No transformation is available between {self.__coords__} and {key}."
                )
        else:
            return self.__XP_reprs__[key]

    def __repr__(self) -> str:
        """String representation of the Trajectory instance"""
        return f"Trajectory with {self.ntraj} particles and {self.nsteps} time slices."

    def __len__(self) -> int:
        """Number of steps in this trajectory"""
        return self.nsteps

    # --- Data manipulation ---

    def _get_cached_scalar(self, key: str) -> torch.Tensor:
        """A helper method to retrieve cached scalar quantities for plotting and analysis."""
            
        match key:
            case "energy":
                q = self.energy
            case "mu_violation":
                q = self.mu_violation
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


    @classmethod
    def join(cls, trajectories: List[Trajectory], fill_reprs: bool = True) -> Trajectory:
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
            Joined trajectory

        """
        if not trajectories:
            return None

        # Compatibility checks
        for t in trajectories:
            ...
            # if t.nsteps != self.nsteps:
            #     raise ValueError(
            #         "Cannot join trajectories with different number of steps."
            #     )
            # if t.__coords__ != self.__coords__:
            #     raise ValueError(
            #         "Cannot join trajectories with different coordinate systems."
            #     )
            # if t.particle_state != self.particle_state:
            #     raise ValueError("Cannot join trajectories with different particle states.")
            # if t.spacetime_state != self.spacetime_state:
            #     raise ValueError("Cannot join trajectories with different spacetime states.")

        all_X = [t.X for t in trajectories]
        all_P = [t.P for t in trajectories]
        all_affine_t = [t.affine_t for t in trajectories]

        all_last_step = [t.last_step for t in trajectories if t.last_step is not None]

        if len(all_last_step) > 0:
            last_step = max(all_last_step)

        all_keys = set()
        for t in trajectories:
            _keys_ = set(t.__XP_reprs__.keys())
            if fill_reprs:
                all_keys.update(_keys_)
            else:
                all_keys.intersection_update(_keys_)

        # for key in all_keys:
        #     new_X, new_P = self.__getitem__(key)
        #     reprs = [t.__getitem__(key) for t in trajectories]
        #     X_ = [new_X] + [_r_[0] for _r_ in reprs]
        #     P_ = [new_P] + [_r_[1] for _r_ in reprs]

        #     new_X = torch.cat(X_)
        #     new_P = torch.cat(P_)

        #     self.__XP_reprs__[key] = new_X, new_P

        # self.ntraj += sum([t.ntraj for t in trajectories])
        # return self

    def join(
        self, trajectories: List[Trajectory], fill_reprs: bool = True
    ) -> Trajectory:
        return self.__class__.join(
            [self].extend(trajectories), fill_reprs=fill_reprs
        )
        
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

        new_X = torch.zeros((self.ntraj, new_nsteps, 4), device=self.X.device)
        new_P = torch.zeros((self.ntraj, new_nsteps, 4), device=self.P.device)

        for i in range(self.ntraj):
            for j in range(4):
                new_X[i, :, j] = torch.interp(
                    new_affine_t, self.affine_t[i], self.X[i, :, j]
                )
                new_P[i, :, j] = torch.interp(
                    new_affine_t, self.affine_t[i], self.P[i, :, j]
                )

        outp = self.__class__(
            X=new_X,
            P=new_P,
            affine_t=new_affine_t,
            particle=self.particle,
            tracer=self.tracer,
            coordinates=self.__coords__,
            last_step=self.last_step,
        )

        return outp

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
            "X": self.X,
            "P": self.P,
            "affine_t": self.affine_t,
            "coord_original": self.__coords__,
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
    def plot2d(
        self,
        projection: Literal['xy', 'yz', 'xz'] | np.ndarray = "xy",
        color_by_value: str = None,
        scope: Tuple[float, float, float, float] = (-20, 20, -20, 20),
        cleaned: bool = False,
        label: Optional[str] = None,
        fig: Optional[plt.Figure] = None,
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

        horizon = 2.0 # Temporary hardcoded horizon radius. # TODO: get horizon slice from spacetime object.

        x, _ = self['Cartesian']
        # x = self.X[..., 1:].numpy()

        q = None
        if color_by_value:
            q = self._get_cached_scalar(color_by_value)

        if cleaned:
            q = bhg.utils._value_cleaning_(q, self._genuine_steps)
            x = bhg.utils._value_cleaning_(x, self._genuine_steps)

        fig, ax = bhg.plot2d(
            x.numpy().swapaxes(0, 1),
            q = q,
            q_label = color_by_value,
            horizon = horizon,
            projection = projection,
            label = label,
            borders=scope,
            fig = fig,
            ax=ax,
            **kwargs
        )

        return fig, ax


    def histogram(
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
        """Plots distribution histogram for given quantities

        Parameters
        ----------

        """
        values = self._get_cached_scalar(quantity)

        if cleaned:
            values = values[self._genuine_steps]

        if replace_nan is not None:
            values.nan_to_num(replace_nan)
        else:
            assert values.isnan().any(), f'Nan values in tensor quantity {quantity}'

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
        dynamic: Literal['time', 'batch'] = 'time',
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
            The quantity to plot. Must be a key for _get_cached_scalar.
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
            Graphic backend to use. ~mpl` is for matplotlib backend - well suited for storing and publishing plots. 
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
        ...
        


    def plot3d(
        self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None
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

        return bhg.plot3d(
            points=self["Cartesian"][0][..., 1:],
            fig=fig,
            ax=ax,
        )

    # TODO: Refactor
    def report(self) -> plt.Figure:
        """Generates a report with several plots of the trajectory.

        The report includes:
        - 2D plot of equatorial motion [optional shift?]
        - 3D plot
        - 3D plot of top-k worst precision trajectories
        - Conservation histogram

        Returns
        -------
        plt.Figure
        """