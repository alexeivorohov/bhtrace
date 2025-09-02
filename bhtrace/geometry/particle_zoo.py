import torch
from .particle import Particle
from .spacetime import Spacetime
from ..functional.diff import Grad


class Photon(Particle):
    '''
    Describes a photon in a given spacetime.
        
    In this class, helicity is not taken into account.
    '''

    def __init__(self, spacetime: Spacetime):
        '''
        ### Inputs:
        - spacetime: Spacetime() - spacetime
        '''
        super().__init__(spacetime=spacetime)
        self.mu = 0
        self.h = None

    
    def Hmlt(self, X, P):

        ginv = self.spacetime.ginv(X)

        return 0.5*torch.einsum('...uv, ...u, ...v -> ...', ginv, P, P)


    def dHmlt(self, X, P, eps):
        # Wrapper for the Hamiltonian to match the signature expected by Grad
        hmlt_func = lambda x: self.Hmlt(x, P)

        # Use 2nd order central difference gradient from functional.diff
        grad_calculator = Grad(hmlt_func, eps=eps, order=2)
        
        return grad_calculator(X)


    def GetNullMomentum(self,
                        X: torch.Tensor,
                        V: torch.Tensor
                        ):
        '''
        Get covariant momentum
        '''
        # E = self.spacetime.tetrad(X)
        gX = self.spacetime.g(X)
        # Step 2: invert tetrad to get e^\mu_a
        # E_inv = torch.inverse(E)  # E = e^a_mu, so E_inv = e^\mu_a^T
        # e_mu_a = E_inv.nT  # shape (4,4)

        # Transform to coordinate basis and lower index, 
        p = torch.einsum('...wu, ...vu, ...v -> ...v', gX, E_inv, V)  # shape (4,)

        # norm_check = torch.einsum.
        # assert torch.allclose()

        return p


    def GetDirection(self, X, P):
        '''
        Inputs:
            X: torch.Tensor - position
            P: torch.Tensor - impulse (downstairs, contravariant)
        '''
        ginvX = self.spacetime.ginv(X)
        v = torch.einsum('...uv, ...u -> v', ginvX, P)
        return v[1:]


    def MomentumNorm(self, X, P):

        ginvX_spatial = self.spacetime.ginv(X)[..., 1:, 1:]
        p2_spatial = torch.einsum('...ij, ...i, ...j ->  ...',
                                  ginvX_spatial, P[..., 1:], P[..., 1:])
        
        P[..., 1:] = P[..., 1:] * torch.pow(p2_spatial, -0.5)
        
        return P
    

    def energy(self, X, P, u):
        '''
        Inputs
        - X: torch.Tensor - point in spacetime
        - P: P_{mu} - 4-impulse of photon 
        - u: u^{mu} 4-velocity of source (medium)
        '''

        return torch.einsum('...u, ...u -> ...', P, u)        


    def normp(self, X, P):

        pass


class EffPhoton(Particle):

    def __init__(self, spacetime: Spacetime):
        '''
        ### Inputs:
        - spacetime: Spacetime() - spacetime
        '''
        super().__init__(spacetime=spacetime)
        self.mu = 0
        self.h = 0 # helicity
        pass


class PhotonR(Photon):

    def __init__(self, spacetime):

        super().__init__(spacetime)
        self.mu = 0


    



