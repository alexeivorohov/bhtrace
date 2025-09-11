import torch
import torch.linalg as LA
from .spacetime import Spacetime


class MinkowskiCart(Spacetime):

    def __init__(self):

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

    def __init__(self, a=0.6, m=1, Q=0):
        
        self.a = a
        self.a2 = a*a
        self.m = m
        self.Q = Q
        self.eta = torch.diag(torch.Tensor([-1, 1, 1, 1]))

        self.cr_r = 0.0


    def g(self, X):

        # X: [bm]
        # Kerr-Newman metric
        a = self.a
        a2 = self.a2
        m = self.m
        Q = self.Q

        shape_a = [*X.shape[:-1]]
        shape_b = [1 for _ in shape_a]
                          
        p = X[..., 1:]
        rho = self.geom_R(X)- a2
        r2 = 0.5*(rho + torch.sqrt(rho**2 + 4.0*a2*p[..., 2]**2))
        r = torch.sqrt(r2)
        self.r = r
        r2a2 = r2 + a2

        k = torch.zeros_like(X)
        k[..., 0] = 1
        k[..., 1] = (r*p[..., 0] + a*p[..., 1])/r2a2
        k[..., 2] = (r*p[..., 1] - a*p[..., 0])/r2a2
        k[..., 3] = p[..., 2]/r

        
        outer_k = torch.einsum('...p, ...q -> ...pq', k, k)
        f = (r2*(2.0*m*r - Q*Q)/(r2*r2+(a*p[..., 2])**2)).view(*shape_a, 1, 1)
        eta = self.eta.view(*shape_b, 4, 4) 

        # print(X.shape)
        # print(outer_k.shape)
        # print(f.shape)
        # print(eta.shape)

        return f*outer_k + eta


    def geom_R(self, X):
        
        return torch.sqrt(torch.einsum('...u, ...u -> ...', X[..., 1:], X[..., 1:]))


    def ginv(self, X):  

        return torch.inverse(self.g(X))


    def crit(self, X):

        p = X[..., 1:]
        rho = self.geom_R(X) - self.a2
        r2 = 0.5*(rho + torch.sqrt(rho**2 + 4.0*self.a2*p[..., 2]**2))
        r = torch.sqrt(r2)

        return r
        

class SchwSchild(Spacetime):

    def __init__(self, m=1.0, Q=0.0):

        self.m = m
        self.Q = Q
        self.Q2 = Q*Q
        self.eta = torch.diag(torch.Tensor([-1, 1, 1, 1]))
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

