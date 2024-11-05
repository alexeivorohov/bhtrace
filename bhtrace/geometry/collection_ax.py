import torch
from .spacetime import Spacetime


class Kerr(Spacetime):

    def __init__(self, a: float):
        '''
        a: float - rotation parameter in units a/M
        '''
        self.a = a
        self.Dlta = lambda r: r**2 - 2*r + a**2
        self.Sgma = lambda r, th: r**2 + a**2 * torch.cos(th)**2
        self.P = lambda r, l: r**2+a**2-a*l
    

    def uR(self, r, l_s, q_s):

        outp = self.P(r, l_s) - self.Dlta(r)*((l_s-a)**2 + q_s**2)

        return outp


    def uTh(self, th, l_s, q_s):

        outp = q_s**2 - torch.cos(th)**2*(-a**2+ (l_s*torch.sin(th))**2)

        return outp
