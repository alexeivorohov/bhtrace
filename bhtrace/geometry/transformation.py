from abc import ABC, abstractmethod
from typing import Tuple, List
import torch

class CoordinateTransformation(ABC):

    def __init__(self, input=None, target=None, inverse=None):

        # TODO: How to manage links with CS?
        self._from_ = input
        self._to_ = target

        self.inverse = inverse


    def __call__(self, X: torch.Tensor):
        '''
        Transform coordinates
        '''

        raise NotImplementedError
    

    def jac(self, X: torch.Tensor):
        '''
        Jacobian of the transformation
        '''

        raise NotImplementedError
    

    def tensor(self,
               X: torch.Tensor,
               A: torch.Tensor,
               valence: List[bool] = None
               ):
        '''
        Transform tensor quantity A

        Inputs:
        - X: torch.Tensor - coordinates (shape[..., 4])
        - A: torch.Tensor - values of the tensor (shape[..., [4]*rank])
        -
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

        # TODO: Implement this method
        raise NotImplementedError
    

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
    
