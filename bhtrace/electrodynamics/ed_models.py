from .electrodynamics import ED_F
import torch

class Maxwell(ED_F):

    def __init__(self):

        # mu_0 = 4*pi
        # eps_0 = 1/(4*pi)
        w = 1/(4*torch.pi)

        super().__init__()
        self.L = lambda F: -w*0.25*F
        self.L_F = lambda F: -w*0.25
        self.L_FF = lambda F: 0
        self.U = lambda X: torch.Tensor([1, 0, 0, 0])

        
        pass



    


        

