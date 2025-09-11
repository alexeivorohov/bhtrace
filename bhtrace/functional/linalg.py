import torch


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