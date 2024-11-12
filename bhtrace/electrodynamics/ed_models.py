from .electrodymamics import ED_F

class Maxwell(ED_F):

    def __init__(self):

        # mu_0 = 4*pi
        # eps_0 = 1/(4*pi)
        self.L = lambda F: 0.25*F
        self.L_F = lambda F: 0.25
        self.L_FF = lambda F: 0
        self.U = torch.Tensor([1, 0, 0, 0])

        super.__init__(L_F, L_FF)
        pass


    def __compute__(self, X):

        self.Tuv = 0.25 * self.L * 

    


        

