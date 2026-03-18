import torch
from itertools import permutations


def levi_civita_tensor(dim):
    '''
    Basic construction of fully-antisymmetric Levi-Civita object in dimension=dim.

    Not invariant form!

    ### Inputs:
    - dim: int - Dimension of the tensor.

    ### Outputs:
    - outp: torch.Tensor - Levi-Civita tensor of shape (dim, dim, ..., dim).
    '''
    outp = torch.zeros((dim,) * dim, dtype=torch.int8)  # Create a dim-dimensional tensor filled with zeros

    # Generate all permutations of dimensions
    for perm in permutations(range(dim)):
        # Calculate the sign of the permutation
        sign = 1
        for i in range(dim):
            for j in range(i + 1, dim):
                if perm[i] > perm[j]:
                    sign *= -1
        outp[perm] = sign  # Assign the sign to the appropriate position in the tensor

    return outp