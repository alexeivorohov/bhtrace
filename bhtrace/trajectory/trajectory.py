from __future__ import annotations
from typing import List, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bhtrace.tracing.tracer import Tracer

import torch

from bhtrace.geometry.spacetime.base import Spacetime
from bhtrace.geometry.particle import Particle
from bhtrace.geometry.transformation import relation_dict

from bhtrace.graphics import Plot2D, PlotValue

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
        return Plot2D.plot_2d(self, ax=ax, figsize=figsize, **kwargs)

    @classmethod
    def plot2d_(cls, trajectories: List[Trajectory], figsize=(10, 10), **kwargs):
        '''
        Plots multiple trajectories on a mosaic of subplots.
        '''
        return Plot2D.plot_2d_mosaic(trajectories, figsize=figsize, **kwargs)
    

    def plot_conservation(self):
        '''
        
        '''
        return PlotValue.plot_conservation(self)
   






