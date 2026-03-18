import pytest
import torch
import unittest.mock as mock
from typing import Tuple

from bhtrace.grrt.runner import GRRT
from bhtrace.grrt.radiation import Blackbody, IntegralFlux
from bhtrace.trajectory.trajectory import Trajectory
from bhtrace.medium._base import Medium
from bhtrace.geometry.spacetime.spherical import MinkowskiSph
from bhtrace.geometry.particle.implementations import Photon
from bhtrace.graphics.report import plot_grrt_report

# Mock Tracer
class MockTracer:
    def __init__(self, particle, spacetime):
        self.particle = particle
        self.spacetime = spacetime

    def state(self):
        return {}

# Mock Volumetric Medium
class VolumetricShell(Medium):
    def __init__(self, spacetime):
        super().__init__(spacetime)
        self.metric = spacetime
        self.r_in = 2.0
        self.r_out = 4.0

    def signed_distance(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 1]
        # This defines a spherical shell between r_in and r_out, with negative values inside.
        return torch.max(r - self.r_out, self.r_in - r)

    def hit_condition(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        # A hit occurs if the new point is inside the medium.
        return s1 <= 0

    def adjust_hit(self, x0: torch.Tensor, x1: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use the midpoint of the segment as a representative point for the radiative calculation.
        return x0 + 0.5 * (x1 - x0), p0 + 0.5 * (p1 - p0)

    def fluid_velocity(self, x: torch.Tensor) -> torch.Tensor:
        # Stationary fluid in the coordinate frame.
        g = self.spacetime.g(x)
        g_tt = g[..., 0, 0]
        u_t = (-g_tt).rsqrt()
        u = torch.zeros_like(x)
        u[..., 0] = u_t
        return u

    # Methods required by the radiative models
    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(1e4, device=x.device, dtype=x.dtype)
    def rest_mass_density(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    def opacity(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    def surface_flux(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(10.0, device=x.device, dtype=x.dtype)

# Pytest Fixtures
@pytest.fixture(scope="module")
def spacetime():
    return MinkowskiSph()

@pytest.fixture(scope="module")
def particle(spacetime):
    return Photon(spacetime)

@pytest.fixture(scope="module")
def tracer(particle):
    return MockTracer(particle, particle.spacetime)

@pytest.fixture
def trajectory_fixture(particle, tracer):
    # A trajectory of 1 ray with 11 steps, moving radially inward.
    n_rays, n_steps = 1, 11
    
    # Position (X) in spherical coordinates
    x = torch.zeros(n_rays, n_steps, 4)
    x[..., 1] = torch.linspace(5, 1, n_steps).unsqueeze(0)  # r: 5 -> 1
    x[..., 2] = torch.pi / 2.0  # Equatorial plane
    
    # Contravariant Momentum (P)
    p = torch.zeros(n_rays, n_steps, 4)
    p[..., 0] = 1.0  # p^t = E
    p[..., 1] = -1.0  # p^r for ingoing radial photon in Minkowski
    
    # Affine parameter (l)
    l = torch.linspace(0, 4, n_steps).unsqueeze(0)
    
    traj = Trajectory(X=x, P=p, particle=particle, tracer=tracer, coordinates="Spherical")
    traj.l = l # Manually attach affine parameter
    
    return traj

@pytest.fixture
def medium_fixture(spacetime):
    return VolumetricShell(spacetime)

# Tests
def test_grrt_runner_spectrum(trajectory_fixture, medium_fixture):
    """Tests the GRRT runner's spectral computation."""
    grrt = GRRT(
        medium=medium_fixture,
        compute_total=False,
        frequences=[1e5, 1e4],
        skip_first=0
    )
    
    bb_model = Blackbody(dtau_thick=1e-2)
    grrt.attach_models(spectral_models=[bb_model])
    
    grrt.compute(trajectory_fixture)
    
    spectrum = grrt.retrieve('spectrum')
    
    assert spectrum is not None
    assert spectrum.shape == (1, len(grrt.frequences))
    assert torch.any(spectrum > 0), "Spectrum should be non-zero after passing through the medium."

def test_grrt_runner_flux(trajectory_fixture, medium_fixture):
    """Tests the GRRT runner's total flux computation."""
    grrt = GRRT(
        medium=medium_fixture,
        compute_total=True,
        skip_first=0
    )
    
    flux_model = IntegralFlux()
    grrt.attach_models(total_models=[flux_model])
    
    grrt.compute(trajectory_fixture)
    
    flux = grrt.retrieve('total')
    
    assert flux is not None
    assert flux.shape == (1,)
    assert torch.all(flux > 0), "Flux should be non-zero after passing through the medium."

def test_grrt_no_hit(trajectory_fixture, medium_fixture):
    """Tests that no radiation is computed when the trajectory does not hit the medium."""
    # Move the medium out of the trajectory's path
    medium_fixture.r_in = 6.0
    medium_fixture.r_out = 8.0
    
    grrt = GRRT(
        medium=medium_fixture,
        compute_total=True,
        frequences=[1e5, 1e6],
        skip_first=0
    )
    
    bb_model = Blackbody(dtau_thick=1e-2)
    flux_model = IntegralFlux()
    grrt.attach_models(total_models=[flux_model], spectral_models=[bb_model])
    
    grrt.compute(trajectory_fixture)
    
    flux = grrt.retrieve('total')
    spectrum = grrt.retrieve('spectrum')
    
    assert torch.all(flux == 0), "Flux should be zero when the medium is not hit."
    assert torch.all(spectrum == 0), "Spectrum should be zero when the medium is not hit."

@mock.patch('matplotlib.pyplot.show')
@mock.patch('matplotlib.pyplot.subplots')
def test_grrt_report(mock_subplots, mock_show, trajectory_fixture, medium_fixture):
    """Tests the GRRT report generation."""
    fig_mock = mock.MagicMock()
    axs_mock = [mock.MagicMock() for _ in range(3)]
    mock_subplots.return_value = (fig_mock, axs_mock)

    grrt = GRRT(
        medium=medium_fixture,
        compute_total=True,
        frequences=[1e5, 1e6],
        skip_first=0,
        probe_idx=0
    )
    
    bb_model = Blackbody(dtau_thick=1e-2)
    flux_model = IntegralFlux()
    grrt.attach_models(total_models=[flux_model], spectral_models=[bb_model])

    grrt.compute(trajectory_fixture)
    fig = plot_grrt_report(grrt, trajectory_fixture)
    
    assert grrt.probe_history is not None
    assert len(grrt.probe_history.x) > 0
    assert mock_subplots.called
    assert fig is fig_mock
