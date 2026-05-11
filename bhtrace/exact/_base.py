from typing import Tuple
from abc import abstractmethod


import torch

class AnalyticSolution:
    """This is a baseclass for all analytic solution generators
    
    """

    @abstractmethod
    def from_cartesian(cls, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> 'AnalyticSolution':
        """Initializes solutions for given initial conditions"""
        pass

    @abstractmethod
    def propagate(self, T: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates coordinates and velocities for given time steps"""
        pass