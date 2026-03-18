from typing import Callable, Literal

import torch

from bhtrace.utils.diff import jacobian

Metric = Callable[[torch.Tensor], torch.Tensor] | torch.Tensor

def numeric_tetrad(
    g: Metric | torch.Tensor,
    x: torch.Tensor,
    method: Literal["gd", "svd"] = "svd",
    **kwargs,
) -> torch.Tensor:
    """
    Numerically computes local tetrad for metric `g` at point `x`.

    Parameters
    ----------
    g : Metric
        A callable that takes a tensor of coordinates (..., 4) and returns
        the metric tensor at that point (..., 4, 4), or the metric tensor itself.
    x : torch.Tensor
        Coordinates tensor, shape (..., 4).
    method : {'svd', 'gd'}, optional
        The method to use for tetrad computation.
        - 'svd': Singular Value Decomposition (faster, more stable).
        - 'gd': Gradient Descent (can be slower, educational).
        Default is 'svd'.
    **kwargs
        Additional keyword arguments to be passed to the chosen method.

    Returns
    -------
    torch.Tensor
        The tetrad matrix E, shape (..., 4, 4), such that g = E^T @ eta @ E.
    """
    if isinstance(g, torch.Tensor):
        g_x = g
    else:
        g_x = g(x)

    match method:
        case 'svd': 
            return _tetrad_svd(g_x, x, **kwargs)
        case 'gd':
            return _tetrad_gd(g_x, x, **kwargs)
        case _:
            raise KeyError(
                f"No method known {method} for numerically tetrad evaluation",
                f"Avaiable methods: {["gd", "svd"]}"
            )


def numeric_conn(
    g: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-5,
    order: int = 2,
) -> torch.Tensor:
    """
    Computes Christoffel symbols from a metric function `g`.
    
    Parameters
    ----------
    g : callable torch.Tensor -> torch.Tensor
        A callable that takes a tensor of coordinates (..., 4)
        and returns the metric tensor at that point (..., 4, 4).
    x: torch.Tensor of shape (..., 4)
        The coordinate tensor.
    eps : float
        The finite difference step size for calculating the derivatives
    order: 
        The order of accuracy for the derivative calculation.

    Returns
    -------
    torch.Tensor The Christoffel symbols Gamma^k_ij at point x, shape (..., 4, 4, 4).
    """
    # jacobian computes d(g_ij)/dx^k and returns it with shape (..., 4, 4, 4)
    # where the last index is `k` (the differentiation variable).
    dg_dx = jacobian(g, x, eps=eps, order=order)  # shape (..., i, j, k)

    ginv = torch.linalg.inv(g(x))

    # Christoffel symbol formula:
    # Gamma^k_ij = 0.5 * g^{kl} * (d_i g_jl + d_j g_il - d_l g_ij)
    # dg_dx has indices (..., i, j, k) for d(g_ij)/dx^k
    # Using einsum for clarity and correctness:
    christoffel = 0.5 * (
        torch.einsum("...kl,...jli->...kij", ginv, dg_dx)  # d_i g_jl
        + torch.einsum("...kl,...ilj->...kij", ginv, dg_dx)  # d_j g_il
        - torch.einsum("...kl,...ijl->...kij", ginv, dg_dx)  # d_l g_ij
    )

    return christoffel

def _tetrad_svd(g: torch.Tensor, x: torch.Tensor, verify=False):
    """
    Finds the tetrad matrix E for a given metric tensor g using SVD.

    The method is based on the fact that any symmetric matrix `g` can be
    diagonalized by an orthogonal matrix of its eigenvectors.

    Parameters
    ----------
    g : torch.Tensor
        Metric tensor at point x, shape (..., 4, 4).
    x : torch.Tensor
        Coordinates tensor, shape (..., 4). Not directly used in computation
        but maintained for API consistency.
    verify : bool, optional
        If True, verifies the decomposition by reconstructing the metric and
        checking for correctness. Default is False.

    Returns
    -------
    torch.Tensor
        The tetrad matrix E of shape (..., 4, 4).
    """

    shape_a = (*[1] * (x.ndim -1), 4, 4)
    shape_b = (*x.shape[:-1], 4, 4)

    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).view(shape_a).to(dtype=x.dtype)

    # Step 1: Diagonalize g with eigen-decomposition
    # g is symmetric, so eigenvalues are real
    eigvals, eigvecs = torch.linalg.eigh(g)  # eigvecs columns are eigenvectors

    # Step 2: Construct sqrt of diagonal eigenvalue matrix with sign correction
    # Since metric has signature (-,+,+,+), eigenvalues can be negative
    # We take sqrt of absolute values and keep sign in sqrt_diag

    sqrt_diag = torch.zeros(shape_b, dtype=x.dtype)
    
    for i in range(4):
        sqrt_diag[..., i, i]= (torch.sign(eigvals[..., i]) * torch.sqrt(torch.abs(eigvals[..., i])))

    # Step 3: Construct tetrad matrix E
    # E = eigvecs @ sqrt_diag @ some Lorentz transform (identity here)
    # This satisfies g = E @ E^T, but we want g = E^T eta E
    # So we rearrange:
    # One way is to find E satisfying E^T eta E = g
    # For simplicity, assume E = sqrtm(g) * L, with L Lorentz transform = I here
    # We approximate E as eigvecs @ sqrt_diag

    _e_i_mu = torch.einsum('...ab, ...bc -> ...ac', eigvecs, sqrt_diag)

    # Verify: compute reconstructed metric
    # g_recon = E.T @ eta @ E

    if verify:
        g_recon = torch.einsum('...ab, ...bc, ... cd -> ...ad', _e_i_mu.swapaxes(-1, -2), eta, _e_i_mu)

        # Check if close to g
        if not torch.allclose(g, g_recon, atol=1e-5):
            print("Warning: Reconstruction error in tetrad factorization.")

    return _e_i_mu

    
def _tetrad_gd(g: torch.Tensor, x: torch.Tensor, lr: float = 1e-4, steps: int = 64, eps: float = 1e-6):
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

    shape_a = (*[1] * (x.ndim-1), 4, 4)
    shape_b = (*x.shape[:-1], 4, 4)

    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).view(shape_a).to(dtype=x.dtype)

    # Initialize E as identity + small noise (requires grad)
    _e_i_mu = torch.eye(4, dtype=x.dtype, device=x.device).repeat(
        *shape_b[:-2], 1, 1
    ) + 0.01 * torch.randn(shape_b, dtype=x.dtype, device=x.device)
    _e_i_mu.requires_grad = True

    optimizer = torch.optim.Adam([_e_i_mu], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        # Compute reconstructed metric: E^T eta E
        g_recon = torch.einsum('...ab, ...bc, ... cd -> ...ad', _e_i_mu.swapaxes(-1, -2), eta, _e_i_mu)

        # Loss: Frobenius norm squared of difference
        loss = torch.norm(g - g_recon, p='fro')
        if loss.item() < eps:
            break

        loss.backward()
        optimizer.step()

    return _e_i_mu.detach()


if __name__ == '__main__':

    x = torch.tensor([0.0, 4.0, torch.pi/2, 2.0])

    g = torch.tensor(
        [
            [-0.5, 0, 0, 0],
            [0, 2.0, 0, 0],
            [0, 0, 16.0, 0,],
            [0, 0, 0, 16.0],
        ]
    )

    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))

    # _e_i_mu = _tetrad_gd(g, x, eps=1e-2, lr=1e-2, steps=128)
    _e_i_mu = _tetrad_svd(g, x)

    g_recon = torch.einsum('...ab, ...bc, ... cd -> ...ad', _e_i_mu.swapaxes(-1, -2), eta, _e_i_mu)

    loss = torch.norm(g - g_recon, p='fro')
    print(loss)

