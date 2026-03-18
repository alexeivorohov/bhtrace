import torch
import pytest

from bhtrace.geometry.particle import Particle, Photon
from bhtrace.geometry.spacetime import (
    MinkowskiCart,
    MinkowskiSph,
    SphericallySymmetric,
    KerrSchild
)


# from bhtrace.functional import sph2cart, cart2sph, points_generate


# [] Should be generalized for other particles and spacetimes
# [] More evaluation points, may be also check spacetime symmetries
SPACETIMES = {
    'minkowski_cart': MinkowskiCart(),
    'minkowski_sph': MinkowskiSph(),
    'spherically_symmetric': SphericallySymmetric(),
    'kerr_schild_01': KerrSchild(a=0.1),
    'kerr_schild_05': KerrSchild(a=0.5),
    'kerr_schild_09': KerrSchild(a=0.9),
}
PHOTONS = {name: Photon(st) for name, st in SPACETIMES.items()}
ATOL = 1e-5
RTOL = 1e-5
EPS = 1e-4


@pytest.mark.parametrize("photon", PHOTONS.values(), ids=PHOTONS.keys())
def test_momentum(photon: Photon):
    """
    Test momentum calculation:
    - norm correctness
    - reconstruction of velocity correctness

    """
    # Points and velocities
    x = torch.tensor([2, 3, 3, 3], dtype=torch.float32)
    v = torch.tensor([0.0, 0.8, 0.6, 0.0])

    # Calculate impulse
    p = photon.null_momentum(x, v)

    # Calculate impulse norm
    ginv = photon.spacetime.ginv(x)
    normP = (ginv @ p) @ p

    # Expected norm
    exp_normP = torch.tensor(0.0, dtype=x.dtype)

    # Reconstruct velocity
    v_ = photon.spatial_velocity(x, p)

    # Test
    assert torch.isclose(normP, exp_normP, atol=10*photon._eps, rtol=10*photon._eps)

    assert torch.allclose(v[1:], v_, atol=10*photon._eps, rtol=10*photon._eps)


@pytest.mark.parametrize("photon", PHOTONS.values(), ids=PHOTONS.keys())
def test_hmlt(photon: Photon):
    """
    Test hamiltonian of particle:
    - Shape correctnes
    - Value correctnes
    """
    # Set up coordinates, velocities and impulses
    x = torch.tensor([2, 3, 3, 3], dtype=torch.float32)
    v_unnorm = torch.tensor([0., 1., 1., 1.], dtype=x.dtype)
    v = v_unnorm / torch.norm(v_unnorm)
    p = photon.null_momentum(x, v)

    # Calculate hamiltonian
    hmlt = photon.hmlt(x, p)

    # Expected hamiltonian
    expH = torch.tensor(0.0, dtype=x.dtype)

    assert hmlt.shape == expH.shape

    assert torch.isclose(hmlt, expH, atol=10*photon._eps, rtol=10*photon._eps)


@pytest.mark.parametrize("photon", PHOTONS.values(), ids=PHOTONS.keys())
def test_diff_hmlt(photon: Photon):
    """
    Test hamiltonian derivative:
    - Shape correctness
    - Check taylor expansion
    """
    x = torch.tensor([0, 10.0, 0, 0]) + torch.randn([12, 4], dtype=torch.float32)
    v_unnorm = torch.randn_like(x, dtype=x.dtype)
    v = v_unnorm / torch.norm(v_unnorm)
    p = photon.null_momentum(x, v)

    d_hmlt = photon.dx_hmlt(x, p)

    # test H(X + dX) approx dH/dX * dX
    dx = torch.randn_like(x) * 1e-5
    hmlt_exp = photon.hmlt(x + dx, p)
    hmlt_approx = torch.einsum('...u, ...u -> ...', d_hmlt, dx)

    recon_err = (hmlt_approx - hmlt_exp).norm()

    assert recon_err < photon._eps,\
        f"Reconstruction error {recon_err:.3e} is greater than numerical precision {photon._eps:.3e}"


# @pytest.mark.parametrize("photon", PHOTONS.values(), ids=PHOTONS.keys())
# def test_save_load(photon):
#     """
#     Test if particle can be saved and loaded
#     """
#     state = photon.state()
#     loaded_photon = Particle.from_dict(state.copy())
#     new_state = loaded_photon.state()

#     # TODO: Fix spacetime serialization
#     state.pop('spacetime')
#     new_state.pop('spacetime')

#     assert state == new_state
