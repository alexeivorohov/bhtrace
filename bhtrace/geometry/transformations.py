from abc import ABC, abstractmethod
from typing import Tuple
import torch


class CoordinateTransformation(ABC):

    def __init__(self, _from_=None, _to_=None, inverse=None):

        # TODO: How to manage links with CS?
        self._from_ = None
        self._to_ = None

        self.inverse = inverse


    def __call__(self, X: torch.Tensor):
        '''
        Transformation
        '''

        raise NotImplementedError
    

    def jac(self, X: torch.Tensor):
        '''
        Jacobian of the transformation
        '''

        raise NotImplementedError
    

    def tensor(self, X: torch.Tensor, A: torch.Tensor, valence=None):
        '''
        Transform tensor quantity A

        Inputs:
        - X: torch.Tensor - coordinates (shape[..., 4])
        - A: torch.Tensor - values of the tensor (shape[..., [4]*rank])

        Valence describes, if tensor dimension is covariant(False) or contravariant(True).
        If None, all valences are assumed to be contravariant.
        '''

        coord_dim = len(X.shape) - 1
        rank = len(A.shape) - coord_dim

        X_new = self.__call__(X)
        
        jac = self.jac(X)
        ijac = None

        if valence == None:
            
            valence = [True] * rank

        assert len(valence) == rank, 'Described valences and tensor rank do not match'

        if not all(valence):
            # inverse jacobian
            ijac = self.inverse.jac(X_new)

        in_idx = [k for k in range(rank)]
        out_idx = [k+rank for k in range(rank)]
        args = [A, [..., *in_idx]]

        for k in range(rank):
            if valence[k]:
                args.append(jac)
            else:
                args.append(ijac)
        
            args.append([..., in_idx[k], out_idx[k]])
        
        args.append(out_idx)
            
        A_new = torch.einsum(*args)
            
        return X_new, A_new
    

class TransformationSequence(CoordinateTransformation):

    def __init__(self, TS: Tuple[CoordinateTransformation, ...], inverse=None):

        self.TS = TS

        if inverse == None:
            invTS = [t.inverse for t in TS]
            invTS = tuple(reversed(invTS))
            self.inverse = TransformationSequence(TS=invTS, inverse = self)
        else:
            self.inverse = inverse

    
    def __call__(self, X: torch.Tensor):

        outX = X
        for t in self.TS:
            outX = t(outX)
        
        return outX
    
    
    def jac(self, X: torch.Tensor):

        raise NotImplementedError
    

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
    

class Sym(CoordinateTransformation):

    # TODO: Implement this class

    def __init__(self, new_labels, old_labels, exprs):


        self.compile()
        pass


    def compile(self):

        pass


    def __call__(self, X: torch.Tensor):
        '''
        Transformation
        '''
        pass


    def jac(self, X: torch.Tensor):

        I = torch.eye(4)

        return I.repeat(*X.shape[:-1], 1, 1)
    
