from .transformation import *
import torch


class Cart2Sph(CoordinateTransformation):

    def __init__(self):
        '''
        Cast coordinates from 4d cartesian to 4d spherical coordinates.

        ### Inputs:
        - inX: torch.Tensor - input coordinates 
        - inP: torch.Tensor - input impulses (or velocities)

        ### Outputs:
        - tuple(outX, outP): torch.Tensor - output coordinates and impulses in spherical coordinates
        '''
        self.inverse = None # Sph2Cart


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

        j = torch.zeros(list(X.shape).append(4))
        
        j[..., 0, 0] = 1

        j[..., 1, 1] = 1
        j[..., 1, 2] = 1
        j[..., 1, 3] = 1

        j[..., 2, 1] = 1
        j[..., 2, 2] = 1
        j[..., 2, 3] = 1

        j[..., 3, 1] = 1
        j[..., 3, 2] = 1
        j[..., 3, 3] = 1

        return j
    

class Sph2Cart(CoordinateTransformation):

    def __init__(self, inverse=None):


        pass

    def __call__(self, X: torch.Tensor):

        
        pass

    def jac(self, X: torch.Tensor):


        pass


class Cart2Ax(CoordinateTransformation):

    def __init__(self, inverse=None):

        if inverse == None:
            super().__init__(inverse = Ax2Cart(inverse=self))
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


class Ax2Cart(CoordinateTransformation):

    def __init__(self, inverse=None):
        
        if inverse == None:
            super().__init__(inverse = Cart2Ax(inverse=self))
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


    def tensor(self, X, A):

        X_new = X + self.pos

        return X_new, A


class Rotation(CoordinateTransformation):

    # TODO: Implement this class

    def __init__(self, pos, inverse=None):
        
        self.pos = pos

        if inverse == None:
            super().__init__(inverse = Shift(pos=-pos, inverse=self))
        else:
            super().__init__(inverse = inverse)    
        
    
    def __call__(self, X):
        
        return X + self.pos
    

    def jac(self, X):

        I = torch.eye(4,4)

        return I.repeat(*X.shape[:-1],1,1)


    def tensor(self, X, A):

        X_new = X + self.pos

        return X_new, A


relation_dict = {

    'Cartesian': {
        'Cartesian': Shift,
        'Spherical': Cart2Sph, 
        'Axial': Cart2Ax,
        'Sym': None
                  }, 

    'Spherical': {
        'Cartesian': Sph2Cart,
        'Spherical': None,
        'Axial': None,
        'Sym': None
                  },

    'Axial': {
        'Cartesian': Ax2Cart,
        'Spherical': None, 
        'Axial': None
        },

    'Sym': None
}