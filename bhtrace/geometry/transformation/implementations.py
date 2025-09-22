from .base import CoordinateTransformation
import torch


class Ident(CoordinateTransformation):

    def __init__(self):
        '''
        Ident transformation
        '''

    def __call__(self, X: torch.Tensor):

        return X
    
    def jac(self, X: torch.Tensor):

        return torch.eye(X.shape[-1]).repeat(*X.shape[:-1], 1, 1)

    def tensor(self, X, A, valence = None):

        return X, A


class Cartesian2Spherical(CoordinateTransformation):

    def __init__(self, inverse=None):
        '''
        Cast coordinates from 4d cartesian to 4d spherical coordinates.

        ### Inputs:
        - inX: torch.Tensor - input coordinates 
        - inP: torch.Tensor - input impulses (or velocities)

        ### Outputs:
        - tuple(outX, outP): torch.Tensor - output coordinates and impulses in spherical coordinates
        '''
        if inverse is None:
            inverse = Spherical2Cartesian(inverse=self)
        super().__init__(inverse=inverse)


    def __call__(self, X: torch.Tensor):
        '''
        Transform vector from cartesian to spherical coordinates
        '''
        x2y2 = X[..., 1]**2 + X[..., 2]**2
        r = torch.sqrt(x2y2 + X[..., 3]**2)
        th = torch.arccos(X[..., 3]/r)
        phi = torch.arctan2(X[..., 2], X[..., 1])

        return torch.stack([X[..., 0], r, th, phi], dim=-1)
    
    
    def jac(self, X: torch.Tensor):
        '''
        Calculate jacobian at a point

        Inputs:
        - X: torch.Tensor - points in Cartesian coordinates
        
        Outputs:
        '''

        j = torch.zeros((*X.shape, 4), device=X.device, dtype=X.dtype)
        
        j[..., 0, 0] = 1.

        x = X[..., 1]
        y = X[..., 2]
        z = X[..., 3]

        r2 = x**2 + y**2 + z**2
        r = torch.sqrt(r2)
        rho2 = x**2 + y**2
        rho = torch.sqrt(rho2)

        # fill jacobian
        # dr/d(x,y,z)
        j[..., 1, 1] = x / r
        j[..., 1, 2] = y / r
        j[..., 1, 3] = z / r

        # d(theta)/d(x,y,z)
        j[..., 2, 1] = (x * z) / (r2 * rho)
        j[..., 2, 2] = (y * z) / (r2 * rho)
        j[..., 2, 3] = -rho / r2

        # d(phi)/d(x,y,z)
        j[..., 3, 1] = -y / rho2
        j[..., 3, 2] = x / rho2
        j[..., 3, 3] = 0.

        # handle poles where rho=0 or r=0
        j[torch.isnan(j)] = 0.
        j[torch.isinf(j)] = 0.

        return j
    

class Spherical2Cartesian(CoordinateTransformation):

    def __init__(self, inverse=None):

        if inverse is None:
            inverse = Cartesian2Spherical(inverse=self)
        super().__init__(inverse=inverse)

    def __call__(self, X: torch.Tensor):
        t = X[..., 0]
        r = X[..., 1]
        th = X[..., 2]
        phi = X[..., 3]

        sin_th = torch.sin(th)
        cos_th = torch.cos(th)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        x = r * sin_th * cos_phi
        y = r * sin_th * sin_phi
        z = r * cos_th

        return torch.stack([t, x, y, z], dim=-1)

    def jac(self, X: torch.Tensor):
        j = torch.zeros((*X.shape[:-1], 4, 4), device=X.device, dtype=X.dtype)

        r = X[..., 1]
        th = X[..., 2]
        phi = X[..., 3]

        sin_th = torch.sin(th)
        cos_th = torch.cos(th)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        j[..., 0, 0] = 1.

        j[..., 1, 1] = sin_th * cos_phi
        j[..., 1, 2] = r * cos_th * cos_phi
        j[..., 1, 3] = -r * sin_th * sin_phi

        j[..., 2, 1] = sin_th * sin_phi
        j[..., 2, 2] = r * cos_th * sin_phi
        j[..., 2, 3] = r * sin_th * cos_phi

        j[..., 3, 1] = cos_th
        j[..., 3, 2] = -r * sin_th
        j[..., 3, 3] = 0.

        return j


class Cartesian2Axial(CoordinateTransformation):

    def __init__(self, inverse=None):

        if inverse == None:
            super().__init__(inverse = Axial2Cartesian(inverse=self))
        else:
            super().__init__(inverse = inverse)


    def __call__(self, X: torch.Tensor):
        '''
        Cast cartesian TXYZ to T rho phi Z
        '''

        x2y2 = X[..., 1]**2 + X[..., 2]**2
        rho = torch.sqrt(x2y2)
        phi = torch.arctan2(X[..., 2], X[..., 1])

        return torch.stack([X[..., 0], rho, phi, X[...,3]], dim=-1)
    

    def jac(self, X: torch.Tensor):
        

        j = torch.zeros(*X.shape, 4)

        r = torch.sqrt(X[..., 1]**2 + X[..., 2]**2)
        cphi = X[..., 1]/r
        sphi = X[..., 2]/r

        j[..., 0, 0] = 1

        j[..., 1, 1] = cphi
        j[..., 1, 2] = - X[..., 2]

        j[..., 2, 1] = sphi
        j[..., 2, 2] = X[..., 1]

        j[..., 3, 3] = 1

        return j


class Axial2Cartesian(CoordinateTransformation):

    def __init__(self, inverse=None):
        
        if inverse == None:
            super().__init__(inverse = Cartesian2Axial(inverse=self))
        else:
            super().__init__(inverse = inverse)

    
    def __call__(self, X: torch.Tensor):

        r, phi = X[..., 1], X[..., 2]
        
        cphi = torch.cos(phi)
        sphi = torch.sin(phi)

        X_new = X.clone().detach()

        X_new[..., 1] = r*cphi
        X_new[..., 2] = r*sphi

        return X_new


    def jac(self, X: torch.Tensor):

        j = torch.zeros(*X.shape, 4)

        r, phi = X[..., 1], X[..., 2]

        cphi = torch.cos(phi)
        sphi = torch.sin(phi)

        j[..., 0, 0] = 1

        j[..., 1, 1] = cphi
        j[..., 1, 2] = sphi

        j[..., 2, 1] = -sphi/r
        j[..., 2, 2] = cphi/r

        j[..., 3, 3] = 1
        
        pass


class Shift(CoordinateTransformation):

    def __init__(self, pos: torch.Tensor, inverse=None):
        
        assert(([*pos.shape] == [4]), 'Position vector is incorrect')

        self.pos = pos

        if inverse == None:
            super().__init__(inverse = Shift(pos=-pos, inverse=self))
        else:
            super().__init__(inverse = inverse)    
        
    
    def __call__(self, X):
        
        return X + self.pos
    

    def jac(self, X):

        I = torch.eye(4)

        return I.repeat(*X.shape[:-1],1,1)

class Rotation(CoordinateTransformation):

    # TODO: Implement this class

    def __init__(self, pos, inverse=None):
        
        self.pos = pos

        if inverse == None:
            super().__init__(inverse = Rotation(pos=pos, inverse=self))
        else:
            super().__init__(inverse = inverse)    
        
    
    def __call__(self, X):
        
        return X + self.pos
    

    def jac(self, X):

        I = torch.eye(4,4)

        return I.repeat(*X.shape[:-1],1,1)
