import torch
import numpy as np
from ..geometry import Spacetime, Particle
from .utils import i2_r

# Base interface:
class RadiativeTransfer(torch.nn.Module):

    def __init__(self):

        super().__init__()


    def term():

        pass


    def forward(traj):

        pass

# Thin disk accretion:
class ThinDisk(RadiativeTransfer):

    def __init__(self, th=0.0, I_r=None):

        super().__init__()
        
        self.set_th(th)

        self.r_isco = 6.0
        f = lambda r: 1 - 2.0/r
        if I_r == None:
            self.I_r = lambda r: i2_r(r, f)


    def set_th(self, th):

        self.norm_v = torch.tensor([np.sin(th), 0, np.cos(th)])
        pass


    def forward(self, particle: Particle, X, Y, Z):

        N_tr, t_ = X.shape[1], X.shape[0]
        norm_v = self.norm_v

        F = torch.zeros(N_tr) # результирующие интенсивности

        # список точек, в которых излучены фотоны, нужен для отладки
        emit = [[] for n in range(N_tr)]

        # Вычислим проекцию на нормаль к диску
        proj = (X*norm_v[0]+Y*norm_v[1]+Z*norm_v[2])

        for n in range(N_tr):
            for t in range(t_-1):
                if (proj[t, n]*proj[t+1, n] < 0):

                    X_ = 0.5*(X[t, n] + X[t+1, n])
                    Y_ = 0.5*(Y[t, n] + Y[t+1, n])
                    Z_ = 0.5*(Z[t, n] + Z[t+1, n])

                    R = torch.sqrt(X_**2 + Y_**2 + Z_**2)

                    F[n] += self.I_r(R)


        return F



