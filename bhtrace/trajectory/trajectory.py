from __future__ import annotations
from typing import List, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bhtrace.tracing.tracer import Tracer

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from bhtrace.geometry.spacetime.base import Spacetime
from bhtrace.geometry.particle import Particle
from bhtrace.functional import opt_mosaic
from bhtrace.geometry.transformation import relation_dict


class Trajectory:

    def __init__(self,
                 X: torch.Tensor,
                 P: torch.Tensor,
                 particle: Particle,
                 tracer: "Tracer",
                 coordinates: str = None,
                 **kwargs):
        """
        A universal class for storing and operating with particle trajectory data.

        Parameters:
        - X: torch.Tensor - Trajectory coordinates.
        - P: torch.Tensor - Trajectory momenta.
        - particle: Particle - The particle that was traced.
        - tracer: "Tracer" - The tracer that was used.
        """

        self.particle: Particle = particle
        '''Particle object for which trajectory was traced'''
        self.particle_state = self.particle.state()

        self.spacetime: Spacetime = particle.spacetime
        '''Spacetime in which trajectory was traced'''
        self.spacetime_state = self.spacetime.state()

        if coordinates == None:
            coordinates = self.spacetime.__coords__
        
        self.__coords__ = coordinates
        '''Coordinates used during tracing'''

        self.tracer: Tracer = tracer
        '''Used tracer object'''
        self.tracer_state = self.tracer.state()

        self.X = X.detach().cpu()
        self.P = P.detach().cpu()
        self.__XP_reprs__ = {}
        '''Dict of Tuple[torch.Tensor, torch.Tensor], which stores\
            different coordinate representations of X and P'''

        self.ntraj = X.shape[0]
        '''Number of trajectories'''

        self.nsteps = X.shape[1]
        '''Number of performed steps'''

        self.last_step = None
        '''Last step, for which not all trajectories were stopped by event condition'''

    def __getitem__(self, key) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def __repr__(self):
        return f"Trajectory with {self.ntraj} particles and {self.nsteps} time slices."
    
    def join(self,
             trajectories: List['Trajectory'],
             fill_reprs: bool = True,
             ) -> 'Trajectory':
        '''
        Joins a list of trajectories to the current one.

        All trajectories must have the same number of steps, same coordinate system,
        and originate from the same particle and spacetime configuration.

        Parameters:
        - trajectories: List['Trajectory'] - A list of Trajectory objects to join.
        - fill_reprs
        
        Returns:
        - self: The modified Trajectory object with joined data.
        '''
        if not trajectories:
            return self

        # Compatibility checks
        for t in trajectories:
            if t.nsteps != self.nsteps:
                raise ValueError("Cannot join trajectories with different number of steps.")
            if t.__coords__ != self.__coords__:
                raise ValueError("Cannot join trajectories with different coordinate systems.")
            if t.particle_state != self.particle_state:
                raise ValueError("Cannot join trajectories with different particle states.")
            if t.spacetime_state != self.spacetime_state:
                raise ValueError("Cannot join trajectories with different spacetime states.")

        all_X = [self.X] + [t.X for t in trajectories]
        all_P = [self.P] + [t.P for t in trajectories]

        all_last_step = [t.last_step for t in trajectories if t.last_step is not None]
        if self.last_step is not None:
            all_last_step.append(self.last_step)
        
        if len(all_last_step) > 0:
            self.last_step = max(all_last_step)

        self.X = torch.cat(all_X, dim=0)
        self.P = torch.cat(all_P, dim=0)

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

    def to(self, device):
        """
        Moves the trajectory data to the specified device.
        """
        self.X = self.X.to(device)
        self.P = self.P.to(device)
        return self
    
    @classmethod
    def from_dict(cls, data):
        """
        Creates a Trajectory object from a dictionary.
        """
        from bhtrace.tracing import MockTracer
        from bhtrace.geometry import Particle

        particle = Particle.from_dict(data['particle_state'])
        spacetime = particle.spacetime
        tracer = MockTracer(particle, spacetime)
        
        traj = cls(data['X'], data['P'], particle, tracer, data['coord_original'])
        if '__XP_reprs__' in data:
            traj.__XP_reprs__ = data['__XP_reprs__']
        return traj
    
    def save(self, filename, save_reprs=True):
        """
        Saves the trajectory data to a file.
        """
        data = {
            'X': self.X,
            'P': self.P,
            'coord_original': self.__coords__,
            'particle_state': self.particle_state,
            'spacetime_state': self.spacetime_state
        }
        if save_reprs:
            data['__XP_reprs__'] = self.__XP_reprs__

        torch.save(data, filename)
    
    @staticmethod
    def load(filename):
        """
        Loads trajectory data from a file.
        """
        data = torch.load(filename)
        return Trajectory.from_dict(data)

    def plot2d(self, ax=None, figsize=(10, 10), **kwargs):
        '''
        Plots 2d trajectories on a given matplotlib axis.
        If no axis is provided, a new figure and axis are created.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        X, P = self['Cartesian']

        circle = patches.Circle((0, 0),  # Center coordinates
                                2.0,  # Radius
                                edgecolor='black',  # Edge color
                                facecolor='black',  # Face color (none for a hollow circle)
                                lw=2)

        ax.plot(X[..., 1].numpy().T, X[..., 2].numpy().T)
        ax.add_patch(circle)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-10, 20])
        ax.set_ylim([-15, 15])

        ax.grid('on')
        ax.set_xlabel('$Y/M$')
        ax.set_ylabel('$Z/M$')

        if ax is not None:
            return fig

    @classmethod
    def plot2d_(cls, trajectories: List[Trajectory], figsize=(10, 10), **kwargs):
        '''
        Plots multiple trajectories on a mosaic of subplots.
        '''
        shape, mosaic = opt_mosaic(trajectories)
        figsize_ = (shape[0] * figsize[0], shape[1] * figsize[1])

        fig, axs = plt.subplot_mosaic(mosaic,
                                      figsize=figsize_)

        for k, traj in enumerate(trajectories):
            ax = axs[k]
            traj.plot2d(ax=ax, **kwargs)
            ax.set_title(k)

        return fig
    

    def plot_conservation(self):
        '''
        
        '''
        vH = torch.log10(torch.abs(self.particle.Hmlt(self.X, self.P) - self.particle.mu)+1e-7)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        ax.plot(vH.detach().cpu().numpy().T)
        ax.set_title('Hamiltonian conservation along trajectory')
        ax.set_xlabel('time step')
        ax.set_ylabel('$\\log_{10} |H - \\mu|$')
        ax.grid(True)

        return fig
   
    def plot_impulses(self, mask=None):
        '''
        
        '''

        labels = ['p0', 'p1', 'p2', 'p3']
        if mask == None:
            mask = torch.ones(self.ntraj, dtype=torch.bool)

        fig, axs = plt.subplots(4, 1, figsize = (15, 20))

        for i, label in enumerate(labels):

            axs[i].plot(self.P[mask, :, i].detach().cpu().T)
            axs[i].set_title(f'{label} along trajectory')
            axs[i].set_ylabel(f'{label}')
            axs[i].set_xlabel('time step')
            axs[i].grid(True)

        return fig
    
    def plot_coords(self, mask=None):

        labels = ['x0', 'x1', 'x2', 'x3']
        if mask == None:
            mask = torch.ones(self.ntraj, dtype=torch.bool)

        fig, axs = plt.subplots(4, 1, figsize = (15, 20))

        for i, label in enumerate(labels):
            
            axs[i].plot(self.X[mask, :, i].detach().cpu().T)
            axs[i].set_title(f'{label} along trajectory')
            axs[i].set_ylabel(f'{label}')
            axs[i].set_xlabel('time step')
            axs[i].grid(True)

        return fig

    def plot_metrics(self, mask=None):
        '''
        Plots 10 independent metric components along the trajectory.
        '''
        if mask is None:
            mask = torch.ones(self.ntraj, dtype=torch.bool)

        g = self.spacetime.g(self.X)
        g = g[mask, :, :, :].detach().cpu()

        labels = []
        components = []
        for i in range(4):
            for j in range(i, 4):
                labels.append(f'g_{i}{j}')
                components.append(g[:, :, i, j])

        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()

        for i, label in enumerate(labels):
            ax = axs[i]
            ax.plot(components[i].numpy().T)
            ax.set_title(f'{label} along trajectory')
            ax.set_ylabel(f'{label}')
            ax.set_xlabel('time step')
            ax.grid(True)
        
        fig.tight_layout()
        return fig

    def plot_quantity(self, func, mask=None, name='Q'):
        '''
        Plots values of a given func(X) over the trajectory.
        func(X) can produce scalar or tensor output.
        '''
        import math

        if mask is None:
            mask = torch.ones(self.ntraj, dtype=torch.bool)

        quantity = func(self.X)
        quantity = quantity[mask, ...].detach().cpu()

        # Flatten tensor components
        if quantity.ndim > 2:
            flat_quantity = quantity.view(*quantity.shape[:2], -1)
        else:
            flat_quantity = quantity.unsqueeze(-1)

        n_components = flat_quantity.shape[2]

        labels = []
        if n_components == 1 and quantity.ndim == 2:
            labels.append(f'{name}')
        else:
            for i in range(n_components):
                labels.append(f'{name}_{i}')
        
        components = [flat_quantity[:, :, i] for i in range(n_components)]

        if n_components == 1:
            nrows, ncols = 1, 1
            figsize = (10, 5)
        else:
            ncols = 2
            nrows = math.ceil(n_components / ncols)
            figsize = (15, 5 * nrows)

        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axs = axs.flatten()

        for i, label in enumerate(labels):
            ax = axs[i]
            ax.plot(components[i].numpy().T)
            ax.set_title(f'{label} along trajectory')
            ax.set_ylabel(f'{label}')
            ax.set_xlabel('time step')
            ax.grid(True)
        
        # Hide unused subplots
        for i in range(n_components, len(axs)):
            axs[i].set_visible(False)

        fig.tight_layout()
        return fig

    def plot3d(self, **kwargs):
        """
        Placeholder for plotting the trajectory.
        """
        raise NotImplementedError("Plotting is not yet implemented.") 
    
    def plot_metric(self, mask=None):

        pass

    def plot3d(self, **kwargs):
        """
        Placeholder for plotting the trajectory.
        """
        raise NotImplementedError("Plotting is not yet implemented.")

    def plot_lensing(self, **kwargs):
        """
        Placeholder for plotting the trajectory.
        """
        raise NotImplementedError("Plotting is not yet implemented.")
    
    def plot_image(self, **kwargs):
        """
        Placeholder for plotting the trajectory.
        """
        raise NotImplementedError("Plotting is not yet implemented.")
