import torch
import torch.linalg as LA
from .spacetime import Spacetime

class MinkowskiCart(Spacetime):

    def __init__(self):
        pass

    def g(self, X):

        outp = torch.zeros([X.shape[0], 4, 4])

        outp[:, 0, 0] = -1
        outp[:, 1, 1] = 1
        outp[:, 2, 2] = 1
        outp[:, 3, 3] = 1

        return outp

    def ginv(self, X):

        outp = torch.zeros([X.shape[0], 4, 4])

        outp[:, 0, 0] = -1
        outp[:, 1, 1] = 1
        outp[:, 2, 2] = 1
        outp[:, 3, 3] = 1

        return outp

    def conn(self, X):

        pass


class KerrSchild(Spacetime):

    def __init__(self, a=0.6, m=1, Q=0):

        self.a = a
        self.m = m
        self.Q = Q


    def g(self, X):

        # X: [bm]
        # Kerr-Newman metric
        a = self.a
        a2 = a*a
        m = self.m
        Q = self.Q

        R2_ = LA.vector_norm(X[:, 1:], dim=1, ord=2) # [b]
        R_ = torch.sqrt(R2_) # [b]

        rho = R2_ - a2  # [b]
        r2 = 0.5*(rho + torch.sqrt(rho**2 + 4.0*a2*X[:,3]**2)) #[b]
        r = torch.sqrt(r2) # [b]
        r2a2 = r2 + a2 # [b]


        k = torch.zeros_like(X) # [bm]
        k[: ,0] = 1 #[b]
        k[: ,1] = (r*X[:, 1] + a*X[:, 2])/r2a2 # [b]
        k[: ,2] = (r*X[:, 2] - a*X[:, 1])/r2a2 # [b]
        k[:, 3] = X[:, 3]/r # [b]

        f = r2*(2.0*m*r - Q*Q)/(r2*r2+(a*X[:, 3])**2) # [b]

        return torch.einsum('b,bi,bj->bij', f, k, k) + torch.diag(torch.Tensor([-1, 1, 1, 1]))


    def ginv(self, X):  

        return torch.inverse(self.g(X))


    def lmbda(self, X):

        a = self.a
        a2 = a*a
        m = self.m
        Q = self.Q

        R2_ = LA.vector_norm(X[:, 1:], dim=1, ord=2) # [b]
        R_ = torch.sqrt(R2_) # [b]

        rho = R2_ - a2  # [b]
        r2 = 0.5*(rho + torch.sqrt(rho**2 + 4.0*a2*X[:,3]**2)) #[b]
        r = torch.sqrt(r2) # [b]

        return r
        

    def conn(self, X):

        pass


class KerrSchild0(Spacetime):

    def __init__(self, a=0.6, m=1, Q=0):

        self.a = a
        self.m = m
        self.Q = Q


    def g(self, X):

        # X: [bm]
        # Kerr-Newman metric
        a = self.a
        a2 = a*a
        m = self.m
        Q = self.Q

        p = X[1:]
        rho = p@p - a2
        r2 = 0.5*(rho + torch.sqrt(rho**2 + 4.0*a2*p[2]**2))
        r = torch.sqrt(r2)
        r2a2 = r2 + a2

        k = torch.zeros(4)
        k[0] = 1
        k[1] = (r*p[0]+a*p[1])/r2a2
        k[2] = (r*p[1]-a*p[0])/r2a2
        k[3] = p[2]/r

        f = r2*(2.0*m*r - Q*Q)/(r2*r2+(a*p[2])**2)

        return f*torch.einsum('i,j->ij', k, k) + torch.diag(torch.Tensor([-1, 1, 1, 1]))


    def ginv(self, X):  

        return torch.inverse(self.g(X))


    def lmbda(self, X):


        pass
        

    def conn(self, X):

        pass


