from .electrodynamics import ED_F
import torch

class Maxwell(ED_F):

    def __init__(self):

        # mu_0 = 4*pi
        # eps_0 = 1/(4*pi)
        w = 1/(16*torch.pi)

        super().__init__()
        self.L = lambda F: -w*F
        self.L_F = lambda F: -w
        self.L_FF = lambda F: 0
        self.U = lambda X: torch.Tensor([1, 0, 0, 0])

        pass


class EulerHeisenberg(ED_F):

    def __init__(self, h):

        # mu_0 = 4*pi
        # eps_0 = 1/(4*pi)
        w = 1/(16*torch.pi)
        h1 = h/(16*torch.pi**2)
        h2 = h1*7/4
        dh1 = 2*h1

        super().__init__()
        self.L = lambda F: -w*F + h1*F**2
        self.L_F = lambda F: -w + dh1*F
        self.L_FF = lambda F: dh1
        self.U = lambda X: torch.Tensor([1, 0, 0, 0])

        pass


class BornInfeld(ED_F): 

    def __init__(self, b):


        pass 


class ModMax(ED_F):

    def __init__(self, gma=0):

        # mu_0 = ?
        # eps_o = ?
        gma = torch.Tensor([gma])
        self.gma = gma
        self.coshgma = torch.cosh(gma)
        self.sinhgma = torch.sinh(gma)
        self.w = 1/(16*torch.pi)*self.coshgma
        self.h = 1/(64*torch.pi)*self.sinhgma

        super().__init__()
        self.L = lambda F: -self.w*F + self.h*abs(F)
        self.L_F = lambda F: -self.w + self.h*torch.sign(F)
        self.L_FF = lambda F: 0

        self.U = lambda X: torch.Tensor([1, 0, 0, 0])

        pass


class Bardeen(ED_F):

    def __init__(self, g=0, m=1):

        self.g = g
        self.g2 = g**2
        self.s = g/m

        super().__init__()
        # неясен правильный коэффициент
        self.l1 = 3/(2*self.s*self.g2)

        self.U = lambda X: torch.Tensor([1, 0, 0, 0])


    def L(self, F):

        x = torch.pow(2* self.g2 * F, -0.5)

        return self.l1 * torch.pow(1+x, -2.5)

    
    def L_F(self, F):

        x = torch.pow(2* self.g2 * F, -0.5)

        return self.l1*1.25*torch.pow(1+x, -3.5)*torch.pow(x, 3)


    def L_FF(self, F):

        x = torch.pow(2* self.g2 * F, -0.5)

        term1 = 1.75

        term2 = - 1.5*(1+x)*torch.pow(x, 2)

        return self.l1*1.25*torch.pow(1+x, -4.5)*torch.pow(x, 3)*(term1 + term2)







        

