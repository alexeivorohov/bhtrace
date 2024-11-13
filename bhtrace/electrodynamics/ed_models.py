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



    


        

