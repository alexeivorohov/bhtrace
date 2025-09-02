import torch

from ..geometry import Spacetime, Particle
from .tracer import Tracer

class CTracer(Tracer):

  def __init__(self, ode_method='RK4'):

    self.name = 'CTracer'
    self.m_param = None
    super().__init__(ode_method=ode_method)


  def __term__(self, 
               t: float, 
               X: torch.Tensor, 
               P: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:

    G_ = self.spc.conn(X)

    dX = P
    # Contract Christoffel symbols: dP_i = - G^j_ik P_j P_k
    dP = -torch.einsum('...jik,...j,...k->...i', G_, P, P)

    return dX, dP
