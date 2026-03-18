import torch
import pytest

from bhtrace.utils.numrel import numeric_tetrad, numeric_conn

# --- Fixtures ---

@pytest.fixture
def minkowski_metric_fn():
    def _minkowski(x: torch.Tensor):
        g = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).to(x.dtype)
        # Add batch dimensions
        return g.view(*([1] * (x.ndim - 1)), 4, 4).expand(*x.shape[:-1], 4, 4)
    return _minkowski

@pytest.fixture
def schwarzschild_metric_fn():
    def _schwarzschild(x: torch.Tensor):
        """
        Schwarzschild metric in spherical coordinates (t, r, theta, phi).
        M is the mass, and we'll use M=1 for simplicity.
        """
        M = 1.0
        r = x[..., 1]
        
        g = torch.zeros(*x.shape[:-1], 4, 4, dtype=x.dtype, device=x.device)
        
        g[..., 0, 0] = -(1 - 2 * M / r)
        g[..., 1, 1] = 1 / (1 - 2 * M / r)
        g[..., 2, 2] = r**2
        g[..., 3, 3] = (r**2) * torch.sin(x[..., 2])**2
        
        return g
    return _schwarzschild


# --- Tests for numeric_tetrad ---

@pytest.mark.parametrize("method", ["svd", "gd"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("input_shape", [[4], [16, 4], [1, 32, 2, 4]])
def test_numeric_tetrad_minkowski(minkowski_metric_fn, method, dtype, input_shape):
    """
    Test numeric_tetrad with Minkowski metric.
    The tetrad should be close to identity (or a Lorentz transformation of it).
    The reconstructed metric should be very close to the original.
    """
    x = torch.randn(input_shape, dtype=dtype)
    g = minkowski_metric_fn(x)
    
    kwargs = {"steps": 1000} if method == "gd" else {}
    tetrad = numeric_tetrad(g, x, method=method, **kwargs)
    
    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).to(dtype).view(*([1] * (x.ndim - 1)), 4, 4)
    g_recon = torch.einsum('...ab,...bc,...cd->...ad', tetrad.transpose(-1, -2), eta, tetrad)

    assert torch.allclose(g, g_recon, atol=1e-5), f"Metric reconstruction failed for method={method}"

@pytest.mark.parametrize("method", ["svd", "gd"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numeric_tetrad_schwarzschild(schwarzschild_metric_fn, method, dtype):
    """
    Test numeric_tetrad with Schwarzschild metric.
    The reconstructed metric should be close to the original.
    """
    x = torch.tensor([0.0, 10.0, torch.pi / 2, 0.0], dtype=dtype) # A point outside the horizon
    g = schwarzschild_metric_fn(x)

    kwargs = {"steps": 500} if method == "gd" else {}
    tetrad = numeric_tetrad(g, x, method=method, **kwargs)

    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).to(dtype)
    g_recon = torch.einsum('...ab,...bc,...cd->...ad', tetrad.transpose(-1, -2), eta, tetrad)

    assert torch.allclose(g, g_recon, atol=1e-4), f"Metric reconstruction failed for method={method}"


# --- Tests for numeric_conn ---

def test_numeric_conn_minkowski(minkowski_metric_fn):
    """
    Test numeric_conn with Minkowski metric.
    All Christoffel symbols should be zero.
    """
    x = torch.randn(1, 4, dtype=torch.float64)
    christoffel = numeric_conn(minkowski_metric_fn, x, order=4)
    
    assert torch.allclose(christoffel, torch.zeros_like(christoffel), atol=1e-9)

def test_numeric_conn_schwarzschild(schwarzschild_metric_fn):
    """
    Test numeric_conn with Schwarzschild metric and compare to known values.
    """
    M = 1.0
    # A point outside the event horizon r > 2M
    x = torch.tensor([0.0, 10.0, torch.pi / 2, 0.0], dtype=torch.float64) 
    r = x[1]
    theta = x[2]

    # Analytical Christoffel symbols for Schwarzschild metric at theta=pi/2
    # Only non-zero symbols at this point:
    expected = torch.zeros(4, 4, 4, dtype=torch.float64)
    expected[0, 1, 0] = expected[0, 0, 1] = M / (r**2 * (1 - 2 * M / r))
    expected[1, 0, 0] = (M / r**2) * (1 - 2 * M / r)
    expected[1, 1, 1] = -M / (r**2 * (1 - 2 * M / r))
    expected[1, 2, 2] = -r * (1 - 2 * M / r)
    expected[1, 3, 3] = -r * (1 - 2 * M / r) * (torch.sin(theta)**2)
    expected[2, 1, 2] = expected[2, 2, 1] = 1 / r
    expected[3, 1, 3] = expected[3, 3, 1] = 1 / r
    expected[3, 2, 3] = expected[3, 3, 2] = 0.0  # cot(pi/2) = 0
    expected[2, 3, 3] = -torch.sin(theta) * torch.cos(theta)  # Should be 0

    # numeric_conn returns Gamma^k_ij, so shape is (k, i, j)
    # Our expected is also (k, i, j)
    result = numeric_conn(schwarzschild_metric_fn, x, order=4, eps=1e-4).squeeze()

    # Check a few values
    assert torch.allclose(result[0, 1, 0], expected[0, 1, 0], atol=1e-5)
    assert torch.allclose(result[1, 0, 0], expected[1, 0, 0], atol=1e-5)
    assert torch.allclose(result[1, 1, 1], expected[1, 1, 1], atol=1e-5)
    assert torch.allclose(result[1, 2, 2], expected[1, 2, 2], atol=1e-5)
    assert torch.allclose(result[2, 1, 2], expected[2, 1, 2], atol=1e-5)
    assert torch.allclose(result[3, 2, 3], expected[3, 2, 3], atol=1e-5)
    assert torch.allclose(result[2, 3, 3], expected[2, 3, 3], atol=1e-5)

    # The cot(theta) term is zero since theta=pi/2, tan(theta) is infinite.
    # sin(theta) = 1 at theta = pi/2
    x_theta_not_pi_2 = torch.tensor([0.0, 10.0, torch.pi / 4, 0.0], dtype=torch.float64)
    r = x_theta_not_pi_2[1]
    theta = x_theta_not_pi_2[2]

    expected_323 = torch.cos(theta) / torch.sin(theta)  # cot(theta)
    
    result2 = numeric_conn(schwarzschild_metric_fn, x_theta_not_pi_2, order=4, eps=1e-4).squeeze()
    assert torch.allclose(result2[3, 2, 3], expected_323, atol=1e-5)
