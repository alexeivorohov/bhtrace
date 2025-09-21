import torch
import torch.linalg as LA
from .spacetime import Spacetime


class MinkowskiCart(Spacetime):

    __coords__ = 'Cartesian'

    def __init__(self):
        super().__init__()


        pass


    def g(self, X):

        outp = torch.zeros(*X.shape[:-1], 4, 4)

        outp[..., 0, 0] = -1
        outp[..., 1, 1] = 1
        outp[..., 2, 2] = 1
        outp[..., 3, 3] = 1

        return outp


    def ginv(self, X):

        outp = torch.zeros(*X.shape[:-1], 4, 4)

        outp[..., 0, 0] = -1
        outp[..., 1, 1] = 1
        outp[..., 2, 2] = 1
        outp[..., 3, 3] = 1

        return outp


    def crit(self, X):

        return torch.ones(*X.shape[:-1])


class KerrSchild(Spacetime):
    """Kerr-Schild metric for a rotating, charged black hole in Cartesian coordinates.

    This class implements the Kerr-Newman metric using Kerr-Schild coordinates,
    which are well-behaved and avoid the coordinate singularity at the event
    horizon.

    Attributes:
        a (float): The spin parameter of the black hole (a/M).
        m (float): The mass of the black hole (M).
        Q (float): The charge of the black hole (Q).
    """

    __coords__ = 'Cartesian'

    def __init__(self, a: float = 0.6, m: float = 1.0, Q: float = 0.0):
        """Initializes the KerrSchild spacetime.

        Args:
            a (float, optional): The spin parameter (a/M). Defaults to 0.6.
            m (float, optional): The mass (M). Defaults to 1.0.
            Q (float, optional): The charge (Q). Defaults to 0.0.
        """
        super().__init__()

        self.a = a
        self.a2 = a*a
        self.m = m
        self.Q = Q
        self.eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))

        self.cr_r = 0.0


    def g(self, X: torch.Tensor) -> torch.Tensor:
        """Calculates the Kerr-Schild metric tensor.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].

        Returns:
            torch.Tensor: The metric tensor `g_uv`, shape [..., 4, 4].
        """
        a = self.a
        a2 = self.a2
        m = self.m
        Q = self.Q

        shape_a = list(X.shape[:-1])
        shape_b = [1] * len(shape_a)
                          
        p = X[..., 1:]
        rho = self.geom_R(X) - a2
        r2 = 0.5 * (rho + torch.sqrt(rho**2 + 4.0 * a2 * p[..., 2]**2))
        r = torch.sqrt(r2)
        self.r = r
        r2a2 = r2 + a2

        k = torch.zeros_like(X)
        k[..., 0] = 1.0
        k[..., 1] = (r * p[..., 0] + a * p[..., 1]) / r2a2
        k[..., 2] = (r * p[..., 1] - a * p[..., 0]) / r2a2
        k[..., 3] = p[..., 2] / r

        outer_k = torch.einsum('...p, ...q -> ...pq', k, k)
        f = (r2 * (2.0 * m * r - Q**2) / (r2 * r2 + (a * p[..., 2])**2)).view(*shape_a, 1, 1)
        eta = self.eta.view(*shape_b, 4, 4).to(X.device)

        return f * outer_k + eta


    def geom_R(self, X: torch.Tensor) -> torch.Tensor:
        """Helper function to compute the spatial radius.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].

        Returns:
            torch.Tensor: The spatial radius for each point.
        """
        return torch.sqrt(torch.einsum('...u, ...u -> ...', X[..., 1:], X[..., 1:]))


    def ginv(self, X: torch.Tensor) -> torch.Tensor:
        """Calculates the inverse Kerr-Schild metric tensor.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].

        Returns:
            torch.Tensor: The inverse metric `g^uv`, shape [..., 4, 4].
        """
        return torch.inverse(self.g(X))


    def crit(self, X: torch.Tensor) -> torch.Tensor:
        """Calculates the Boyer-Lindquist radius `r` as the criticality criterion.

        Args:
            X (torch.Tensor): Spacetime coordinates, shape [..., 4].

        Returns:
            torch.Tensor: The value of the Boyer-Lindquist radius `r`.
        """
        p = X[..., 1:]
        rho = self.geom_R(X) - self.a2
        r2 = 0.5 * (rho + torch.sqrt(rho**2 + 4.0 * self.a2 * p[..., 2]**2))
        r = torch.sqrt(r2)

        return r
        

class SchwSchild(Spacetime):

    __coords__ = 'Cartesian'

    def __init__(self, m=1.0, Q=0.0):
        super().__init__()

        self.m = m
        self.Q = Q
        self.Q2 = Q*Q
        self.eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
        self.cr_r = 0.0


    def g(self, X):

        m = self.m
        Q = self.Q

        shape_a = [*X.shape[:-1]]
        shape_b = [1 for _ in shape_a]
                          
        p = X[..., 1:]
        rho = self.geom_R(X) 
        r2 = 0.5*(rho + torch.sqrt(rho**2 ))
        r = torch.sqrt(r2)
        self.r = r
        r2a2 = r2

        k = torch.zeros_like(X)
        k[..., 0] = 1
        k[..., 1] = (r*p[..., 0])/r2a2
        k[..., 2] = (r*p[..., 1])/r2a2
        k[..., 3] = p[..., 2]/r

        
        outer_k = torch.einsum('...p, ...q -> ...pq', k, k)
        f = (r2*(2.0*m*r - Q*Q)/(r2*r2)).view(*shape_a, 1, 1)
        eta = self.eta.view(*shape_b, 4, 4) 

        # print(X.shape)
        # print(outer_k.shape)
        # print(f.shape)
        # print(eta.shape)

        return f*outer_k + eta


    # def conn(self, X):

    #     # TODO: Implement this

    #     pass


    def geom_R(self, X):
        
        return torch.einsum('...u, ...u -> ...', X[..., 1:], X[..., 1:])


    def ginv(self, X):  

        return torch.inverse(self.g(X))


    def crit(self, X):

        p = X[..., 1:]
        rho = self.geom_R(X)
        r2 = 0.5*(rho + torch.sqrt(rho**2))
        r = torch.sqrt(r2)

        return r



    def crit(self, X):

        p = X[..., 1:]
        rho = self.geom_R(X)
        r2 = 0.5*(rho + torch.sqrt(rho**2))
        r = torch.sqrt(r2)

        return r

