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

        # --- Shape checks ---
        assert X.shape[-1] == 4, 'Coordinate dimension must be 4'
        if rank > 0:
            assert A.shape[-rank:] == tuple([4]*rank), 'Tensor dimensions must be 4'
        assert X.shape[:-1] == A.shape[:-rank], 'Batch dimensions of coordinates and tensor do not match'
        # --- End of shape checks ---

        X_new = self.__call__(X)
        
        if rank == 0:
            return X_new, A # Scalar does not transform

        # jac is J^new_old = dx'/dx, ijac is (J^-1)^old_new = dx/dx'
        jac = self.jac(X)
        ijac = None

        if valence == None:
            valence = [True] * rank

        assert len(valence) == rank, 'Described valences and tensor rank do not match'

        if not all(valence):
            # inverse jacobian is needed for covariant components
            ijac = self.inverse.jac(X_new)

        in_idx = list(range(rank))
        out_idx = list(range(rank, 2*rank))
        # This einsum call is equivalent to building a string like '...ab,...ca,...bd->...cd'
        # for a (1,1) tensor, but avoids string manipulation.
        args = [A, [..., *in_idx]]

        for k in range(rank):
            if valence[k]:
                # Contravariant: A'^k = J^k_i A^i. einsum: ...i,...ki->...k
                args.append(jac)
                args.append([..., out_idx[k], in_idx[k]])
            else:
                # Covariant: A'_l = (J^-1)^j_l A_j. einsum: ...j,...jl->...l
                args.append(ijac)
                args.append([..., in_idx[k], out_idx[k]])
        
        args.append([..., *out_idx])
        # print(*args)
            
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
    
