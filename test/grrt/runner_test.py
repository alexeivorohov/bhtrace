import pytest
import torch
import unittest.mock as mock
from typing import Tuple


from bhtrace import Trajectory
from bhtrace.scenarios.makers import make_schwarzschild, make_kerr

# import bhtrace.geometry.spacetime as st

# from bhtrace.graphics.report import plot_grrt_report

from bhtrace.grrt.runner import GRRT
from bhtrace.grrt.radiation import Blackbody, IntegralFlux
from bhtrace.medium import VolumetricShell

TEST_TRAJECTORIES = [
    make_schwarzschild(),
    make_kerr(),
]


def test_runner_step():
    ...


# @pytest.fixture
# def trajectory_fixture(particle, tracer):
#     # A trajectory of 1 ray with 11 steps, moving radially inward.
#     n_rays, n_steps = 1, 11
    
#     # Position (X) in spherical coordinates
#     x = torch.zeros(n_rays, n_steps, 4)
#     x[..., 1] = torch.linspace(5, 1, n_steps).unsqueeze(0)  # r: 5 -> 1
#     x[..., 2] = torch.pi / 2.0  # Equatorial plane
    
#     # Contravariant Momentum (P)
#     p = torch.zeros(n_rays, n_steps, 4)
#     p[..., 0] = 1.0  # p^t = E
#     p[..., 1] = -1.0  # p^r for ingoing radial photon in Minkowski
    
#     # Affine parameter (l)
#     l = torch.linspace(0, 4, n_steps).unsqueeze(0)
    
#     traj = Trajectory(X=x, P=p, particle=particle, tracer=tracer, coordinates="Spherical")
#     traj.l = l # Manually attach affine parameter
    
#     return traj

# @pytest.fixture
# def medium_fixture(spacetime):
#     return VolumetricShell(spacetime)

# # Tests
# def test_grrt_runner_spectrum(trajectory_fixture, medium_fixture):
#     """Tests the GRRT runner's spectral computation."""
#     grrt = GRRT(
#         medium=medium_fixture,
#         compute_total=False,
#         frequences=[1e5, 1e4],
#         skip_first=0
#     )
    
#     bb_model = Blackbody(dtau_thick=1e-2)
#     grrt.attach_models(spectral_models=[bb_model])
    
#     grrt.compute(trajectory_fixture)
    
#     spectrum = grrt.retrieve('spectrum')
    
#     assert spectrum is not None
#     assert spectrum.shape == (1, len(grrt.frequences))
#     assert torch.any(spectrum > 0), "Spectrum should be non-zero after passing through the medium."

# def test_grrt_runner_flux(trajectory_fixture, medium_fixture):
#     """Tests the GRRT runner's total flux computation."""
#     grrt = GRRT(
#         medium=medium_fixture,
#         compute_total=True,
#         skip_first=0
#     )
    
#     flux_model = IntegralFlux()
#     grrt.attach_models(total_models=[flux_model])
    
#     grrt.compute(trajectory_fixture)
    
#     flux = grrt.retrieve('total')
    
#     assert flux is not None
#     assert flux.shape == (1,)
#     assert torch.all(flux > 0), "Flux should be non-zero after passing through the medium."

# def test_grrt_no_hit(trajectory_fixture, medium_fixture):
#     """Tests that no radiation is computed when the trajectory does not hit the medium."""

#     # use magicmock: substitute `is_hit` outputs to return false bool tensor

#     grrt = GRRT(
#         medium=medium_fixture,
#         compute_total=True,
#         frequences=[1e5, 1e6],
#         skip_first=0
#     )
    
#     bb_model = Blackbody(dtau_thick=1e-2)
#     flux_model = IntegralFlux()
#     grrt.attach_models([flux_model, bb_model])  
#     data = grrt.compute(trajectory_fixture)

    
#     assert torch.all(flux == 0), "Flux should be zero when the medium is not hit."
#     assert torch.all(spectrum == 0), "Spectrum should be zero when the medium is not hit."
