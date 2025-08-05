'''
This file contains an abstract class Spacetime, which calculates the principal
quantities associated with a given spacetime: the metric and the Levi-Civita connection
symbols. The abstract class Spacetime contains the following methods:

    g(self, X): 
        Abstract method for calculating the metric at a given point X
        (or at a batch of points, feature under development).

    ginv(self, X):
        Abstract method for calculating the inverse metric at a given point X
        (with raised indices).

    dg(self, X):
        Calculates numerically the (non-)tensor of partial derivatives of the metric 
        specified in the method g(X) at the point X. Uses the simplest first-oder
        difference scheme.

    dg_horder(self, X):
        Calculates numerically the (non-)tensor of partial derivatives of the metric 
        specified in the method g(X) at the point X. Uses order 2 or order 4 difference
        scheme for calculating derivatives.

    conn(self, X):
        Abstract method. Computes the Levi-Civita connection coefficients 
        (Christoffel symbols) via a method specified down the line.

    conn_(self, X, method='standard'):
        Computes the Levi-Civita connection coefficients numerically
        using either dg ('standard') or dg_horder ('horder') method. 

    crit(self, X):
        Abstract method. Computes the "proximity" value to 
        the singularity of the metric.
'''

from abc import ABC, abstractmethod

# from typing_extensions import ParamSpecArgs

import torch

# TODO: May be derivatives and connections should be moved to another class

class Spacetime(ABC):
    '''
    Base class for handling different spacetimes.

    Required methods:
    - g(X): expression for metric function
    - ginv(X): expression for inverse metric function
    
    Optional methods:
    - conn(X): Connection symbols Gamma^p_uv

    '''

    @abstractmethod
    def g(self, X):
        '''
        Metric tensor evaluated at a batch of coordinates

        ## Input
        X: torch.Tensor [..., 4] - coordinates

        ## Output
        g: torch.Tensor [..., 4, 4] - metric tensor at each coordinate
        '''
        pass

    @abstractmethod
    def ginv(self, X):
        '''
        Metric inverse evaluated at a batch of coordinates

        ## Input
        X: torch.Tensor [..., 4] - coordinates

        ## Output
        g: torch.Tensor [..., 4, 4] - metric inverse at each coordinate
        '''

        pass


    def dg(self, X, eps=2e-5) -> torch.Tensor:
        '''
        Numerical derviative of the metric

        ## Input:
        - X: torch.Tensor of shape [..., 4] - point for which to evaluate
        - eps: float (2e-5 default)

        ## Output

        - dg: torch.Tensor of shape [..., 4, 4, 4]
        '''

        gX = self.g(X)
        dgX = torch.zeros(*X.shape[:-1], 4, 4, 4)

        dVec = torch.eye(4).repeat(*X.shape[:-1], 1, 1) * eps


        dgX[..., 0, :, :] = (self.g(X + dVec[..., 0, :]) - gX) / eps
        dgX[..., 1, :, :] = (self.g(X + dVec[..., 1, :]) - gX) / eps
        dgX[..., 2, :, :] = (self.g(X + dVec[..., 2, :]) - gX) / eps
        dgX[..., 3, :, :] = (self.g(X + dVec[..., 3, :]) - gX) / eps

        return dgX


    def conn(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Computes the Levi-Civita connection coefficients (Christoffel symbols) 
        via a method specified down the line.
        
        ### Inputs:
        - X: torch.Tensor [..., 4] - evaluation point(s)

        ### Outputs:
        - G: torch.Tensor [..., 4, 4, 4] - connection symbols
        First index is contravariant, the other two are covariant.
        '''

        return self.conn_(X, method='standard')


    def conn_(self, X, method='standard'):
        '''
        Evaluate connection symbols by numerical differentiation. 
        Relies on method dg(X) in computing derivatives of the metric

        ### Inputs:
        - X: torch.Tensor [4] - coordinates
        ### Outputs:
        - G: torch.Tensor [4, 4, 4] - connection symbols
        First index is contravariant, others are covariant.
        '''

        if method == 'standard':
            g_duv = self.dg(X)
            ginv_ = self.ginv(X)
            
        elif method == 'horder':
            g_duv = self.gd_horder(X)
            ginv_ = self.ginv(X)

        dg0 = torch.einsum('...md, ...duv ->...muv', ginv_, g_duv)
        dg1 = torch.einsum('...mv,...duv -> ...mdv', ginv_, g_duv)
        dg2 = torch.einsum('...mu, ...duv -> ...mud', ginv_, g_duv)
        
        return 0.5*( - dg0 + dg1 + dg2)
    
    def tetrad(self, X: torch.Tensor, method='gd'):

        if method == 'gd':
            return self.tetrad_gd(X)
        else:
            return self.tetrad_linalg(X)


    def tetrad_linalg(self, X : torch.Tensor, verify=False):
        '''
        Given a coordinate4x4 metric tensor g (symmetric), find tetrad matrix E such that
        g = E^T @ eta @ E (index positions arranged accordingly).

        Args:
            X: torch.Tensor of shape (...,4), position in spacetime

        Returns:
            E: torch.Tensor of shape (4,4), tetrad matrix e^a_mu,
            satisfying g_{mu nu} = eta_{ab} e^a_mu e^b_nu
        '''

        g = self.g(X)

        shape_a = (*[1] * (X.ndim -1), 4, 4)

        shape_b = (*X.shape[:-1], 4, 4)

        eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).view(shape_a).to(dtype=X.dtype)

        # Step 1: Diagonalize g with eigen-decomposition
        # g is symmetric, so eigenvalues are real
        eigvals, eigvecs = torch.linalg.eigh(g)  # eigvecs columns are eigenvectors

        # Step 2: Construct sqrt of diagonal eigenvalue matrix with sign correction
        # Since metric has signature (-,+,+,+), eigenvalues can be negative
        # We take sqrt of absolute values and keep sign in sqrt_diag

        sqrt_diag = torch.zeros(shape_b)
        
        for i in range(4):
            sqrt_diag[..., i, i]= (torch.sign(eigvals[..., i]) * torch.sqrt(torch.abs(eigvals[..., i])))

        # Step 3: Construct tetrad matrix E
        # E = eigvecs @ sqrt_diag @ some Lorentz transform (identity here)
        # This satisfies g = E @ E^T, but we want g = E^T eta E
        # So we rearrange:
        # One way is to find E satisfying E^T eta E = g
        # For simplicity, assume E = sqrtm(g) * L, with L Lorentz transform = I here
        # We approximate E as eigvecs @ sqrt_diag

        E = torch.einsum('...ab, ...bc -> ...ac', eigvecs, sqrt_diag)

        # Verify: compute reconstructed metric
        # g_recon = E.T @ eta @ E

        if verify:
            g_recon = torch.einsum('...ab, ...bc, ... cd -> ...ad', E.swapaxes(-1, -2), eta, E)

            # Check if close to g
            if not torch.allclose(g, g_recon, atol=1e-5):
                print("Warning: Reconstruction error in tetrad factorization.")

        return E
    
    
    def tetrad_gd(self, X, lr=1e-2, steps=200, verbose=False):
        """
        Find tetrad matrix E (4x4) such that g = E^T eta E by gradient descent.

        Args:
            X: position in spacetime
            lr: learning rate for optimizer.
            steps: number of gradient descent steps.
            verbose: print loss during training.

        Returns:
            E: torch.Tensor (4,4), tetrad matrix.
        """

        gX = self.g(X)
        shape_a = (*[1] * (X.ndim -1), 4, 4)
        shape_b = (*X.shape[:-1], 4, 4)

        eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).view(shape_a).to(dtype=X.dtype)

        # Initialize E as identity + small noise (requires grad)
        E = torch.eye(4).repeat(*shape_b[:-2], 1, 1) + 0.01 * torch.randn(shape_b)
        E.requires_grad = True

        optimizer = torch.optim.Adam([E], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()

            # Compute reconstructed metric: E^T eta E
            g_recon = torch.einsum('...ab, ...bc, ... cd -> ...ad', E.swapaxes(-1, -2), eta, E)

            # Loss: Frobenius norm squared of difference
            loss = torch.norm(gX - g_recon, p='fro')**2

            loss.backward()
            optimizer.step()

            if (step == steps - 1):
                print(f"Step {step:4d}: loss = {loss.item():.6e}")

        return E.detach()



    @abstractmethod
    def crit(self, X: torch.Tensor):
        '''
        Function of "distance" to the metric singularities
        F: X -> (0, inf)

        Used to control step size or stop integration.
        '''

        return None
    

    def compile(self):
        '''
        Compile the class with torch.jit.script
        '''

        return torch.jit.script(self)


    def __str__(self):
        return self.__class__.__name__
    

class mock_spacetime(Spacetime):
    
    def __init__(self, coefs=[1.0, 2.0, 3.0, 5.0]):
        '''
        :class:`Spacetime()` implementation, used for test purposes.
        '''

        self.coefs = coefs
        pass


    def g(self, X):
        
        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = - self.coefs[0]
        outp[..., 1, 1] = self.coefs[1]
        outp[..., 2, 2] = self.coefs[2]
        outp[..., 3, 3] = self.coefs[3]

        return outp


    def ginv(self, X):

        outp = torch.zeros(*X.shape, 4)
        outp[..., 0, 0] = - 1/self.coefs[0]
        outp[..., 1, 1] = 1/self.coefs[1]
        outp[..., 2, 2] = 1/self.coefs[2]
        outp[..., 3, 3] = 1/self.coefs[3]

        return outp

    
    def crit(self, X):

        outp = abs(X[...,1]-6) + abs(X[...,2]-6) + abs(X[...,3]-6)

        return outp
