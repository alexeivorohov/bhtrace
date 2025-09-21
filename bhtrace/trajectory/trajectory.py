from __future__ import annotations
from typing import List, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bhtrace.tracing.tracer import Tracer

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from bhtrace.geometry import Spacetime, Particle
from bhtrace.functional import opt_mosaic
from bhtrace.geometry.transformation_collection import relation_dict


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

        self.particle = particle
        self.particle_state = self.particle.state()

        self.spacetime: Spacetime = particle.spacetime
        self.spacetime_state = self.spacetime.state()

        if coordinates == None:
            coordinates = self.spacetime.__coords__
        
        self.__coords__ = coordinates

        self.tracer = tracer
        self.tracer_state = self.tracer.state()

        self.X = X.detach().cpu()
        self.P = P.detach().cpu()
        self.R = None
        self.__XP_reprs__ = {}

        self.ntraj = X.shape[0]
        self.nsteps = X.shape[1]
        self.last_step = None

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
    
    def append(self):
        '''
        Placeholder for trajectory joining method
        '''
        return NotImplementedError

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
        ax.set_title('Hamiltonian consservation along trajectory')
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
