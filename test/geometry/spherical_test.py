import torch
import pytest

from bhtrace.geometry import Spacetime
from bhtrace.geometry.spacetime.spherical import (
    MinkowskiSph,
    SphericallySymmetric,
    KerrBL,
    KerrNewmanBL,
)

# Collection of spacetimes to test
SPACETIMES = [
    # MinkowskiSph(),
    # SphericallySymmetric(),  # Schwarzschild
    # SphericallySymmetric(A=lambda r: 1 - 2 / r + r**-2, A_r=lambda r: 2 * r**-2 - 2 * r**-3),  # Reissner-Nordstrom with M=Q=1
    KerrBL(a=0.9),
    KerrBL(a=0.0),  # Schwarzschild
    # KerrNewmanBL(a=0.9, q=0.3),
    # KerrNewmanBL(a=0.0, q=0.0),  # Schwarzschild
]

DEVICES = ['cpu']
# if torch.cuda.is_available():
#     DEVICES.append('cuda:0')

DTYPES = [
    # torch.float32, 
    torch.float64,
]

def fetch_coords(n_eval=10, r_range=(10, 20.0)):
    # Avoid coordinate singularities and event horizon
    coords = torch.rand(n_eval, 4)
    coords[:, 1] = coords[:, 1] * (r_range[1] - r_range[0]) + r_range[0]  # r
    coords[:, 2] = (
        coords[:, 2] * (0.9 * torch.pi) + 0.05 * torch.pi
    )  # theta, away from 0 and pi
    coords[:, 3] = coords[:, 3] * 2 * torch.pi  # phi
    return coords

N_EVAL = 4
x_far_uncast = fetch_coords(N_EVAL, (10, 20))
x_near_uncast = fetch_coords(N_EVAL, (6, 12))
x_close_uncast = fetch_coords(N_EVAL, (4, 8))
x_coords_uncast = [
    x_far_uncast, 
    # x_near_uncast, 
    # x_close_uncast
]

NUMERIC_TOLERANCES = [

]

ANALYTIC_TOLERANCES = [

]

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("x_uncast", x_coords_uncast)
@pytest.mark.parametrize("st", SPACETIMES)
def test_metric(x_uncast: torch.Tensor, st: Spacetime, dtype: torch.dtype, device: str):
    """Test if g @ ginv = I."""
    x = x_uncast.to(dtype=dtype, device=device)

    g = st.g(x)
    assert g.dtype == dtype, f'`g` dtype {g.dtype} not matches test dtype {dtype}'
    assert str(g.device) == device, f'`g` device {g.device} not matches test device {device}'
    ginv = st.ginv(x)
    assert ginv.dtype == dtype, f'`ginv` dtype {ginv.dtype} not matches test dtype {dtype}'
    assert str(ginv.device) == device, f'`ginv` device {ginv.device} not matches test device {device}'
    identity = torch.eye(4, dtype=x.dtype, device=x.device)
    gginv = g.bmm(ginv)
    
    tol = 1e-4 if dtype == torch.float64 else 1e-4
    allclose = torch.allclose(gginv, identity.expand_as(gginv), atol=tol, rtol=tol)

    err = (gginv - identity.expand_as(gginv))
    
    assert allclose, (
        f"Metric inverse check failed for {st.__class__.__name__} with dtype {dtype} on {device}\n"
        f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e} "
    )
# @pytest.mark.skip()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", [1e-1, 1e-2, 3e-3, 1e-4])
@pytest.mark.parametrize("x_uncast", x_coords_uncast)
@pytest.mark.parametrize("st", SPACETIMES)
def test_conn_vs_numerical(x_uncast: torch.Tensor, st: Spacetime, eps: float, dtype: torch.dtype, device: str):
    """Test if analytic connection matches numerical."""

    if not (hasattr(st, "__analytic_conn__") and st._has_analytic_conn):
        pytest.skip(f"{st.__class__.__name__} does not have an analytic connection.")

    if dtype == torch.float32 and eps < 1e-2:
        pytest.skip("High precision test for float32 is unstable")

    x = x_uncast.to(dtype=dtype, device=device)
    st._eps = eps

    conn_analytic = st.conn(x)
    conn_numerical = st.conn_(x)

    tol = 10 * st._eps
    allclose = torch.allclose(conn_analytic, conn_numerical, atol=tol, rtol=tol)

    err = conn_analytic - conn_numerical

    assert allclose, (
        f"Connection vs numerical failed for {st.__class__.__name__} with dtype {dtype} on {device} and eps {eps:.1e}\n"
        f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e} "
    )

# @pytest.mark.skip()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", ANALYTIC_TOLERANCES)
@pytest.mark.parametrize("x_uncast", x_coords_uncast)
@pytest.mark.parametrize("st", SPACETIMES)
def test_analytic_tetrad(x_uncast: torch.Tensor, st: Spacetime, eps: float, dtype: torch.dtype, device: str):
    """Test if tetrad reconstructs the inverse metric."""
    x = x_uncast.to(dtype=dtype, device=device)
    try:
        e_i_mu = st.tetrad(x)
    except (NotImplementedError, AttributeError):
        pytest.skip(f"Tetrad not implemented for {st.__class__.__name__}")
        return
    assert e_i_mu is not None,(
        f"Tetrad returned none for {st.__class__.__name__}"
    )
    nans = e_i_mu.isnan()
    assert not torch.any(nans),(
        f"Tetrad returned nans for samples {nans.any(dim=-1).any(dim=-1)}"
    )

    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=x.dtype, device=x.device))
    ginv = st.ginv(x)

    ginv_recon = torch.einsum("...ia, ab,...jb->...ij", e_i_mu, eta, e_i_mu)

    allclose = torch.allclose(ginv, ginv_recon, atol=eps*10, rtol=eps*10)

    err = (ginv-ginv_recon)

    assert allclose, (
        f"Tetrad reconstruction of inverse metric failed for {st.__class__.__name__} with dtype {dtype} on {device}\n"
        f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e} "
    )

@pytest.mark.skip()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5])
@pytest.mark.parametrize("x_uncast", x_coords_uncast)
@pytest.mark.parametrize("st", SPACETIMES)
def test_taylor(x_uncast: torch.Tensor, st: Spacetime, eps: float, dtype: torch.dtype, device: str):
    """Checks if metric at a nearby point can be approximated by a Taylor expansion using `dg`."""
    if dtype == torch.float32 and eps < 1e-4:
        pytest.skip("High precision test for float32 is unstable")

    x = x_uncast.to(dtype=dtype, device=device)
    st._eps = 1e-5 if dtype == torch.float64 else 1e-3

    dx = torch.randn_like(x)
    dx = dx / torch.linalg.norm(dx, dim=-1, keepdim=True) * eps

    g_x = st.g(x)
    dg_x = st.dg(x)
    g_x_plus_dx = st.g(x + dx)

    g_taylor = g_x + torch.einsum("...puv,...p->...uv", dg_x, dx)

    tol = 100 * eps**2
    allclose = torch.allclose(g_x_plus_dx, g_taylor, atol=tol, rtol=tol)

    err = g_x_plus_dx - g_taylor

    assert allclose, (
        f"Taylor expansion failed for {st.__class__.__name__} with eps={eps:.1e}, dtype {dtype} on {device}\n"
        f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e}"
    )
# @pytest.mark.skip()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", [1e-1, 1e-2, 3e-3, 1e-4])
@pytest.mark.parametrize("x_uncast", x_coords_uncast)
@pytest.mark.parametrize("st", SPACETIMES)
def test_lnrf(x_uncast: torch.Tensor, st: Spacetime, eps: float, dtype: torch.dtype, device: str):
    """Test if LNRF tetrad can be computed and reconstructs the inverse metric."""
    x = x_uncast.to(dtype=dtype, device=device)
    try:
        e_i_mu_lnrf = st.lnrf(x)
    except (NotImplementedError, AttributeError):
        pytest.skip(f"LNRF not implemented for {st.__class__.__name__}")
        return

    assert e_i_mu_lnrf is not None, (
        f"LNRF tetrad returned none for {st.__class__.__name__}"
    )
    nans = e_i_mu_lnrf.isnan()
    assert not torch.any(nans), (
        f"LNRF tetrad returned nans for samples {nans.any(dim=-1).any(dim=-1)}"
    )

    st._eps = eps
    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=x.dtype, device=x.device))
    ginv = st.ginv(x)

    ginv_recon_lnrf = torch.einsum("...ia, ab,...jb->...ij", e_i_mu_lnrf, eta, e_i_mu_lnrf)

    tol = 10 * st._eps
    allclose_recon = torch.allclose(ginv, ginv_recon_lnrf, atol=tol, rtol=tol)

    err_recon = ginv - ginv_recon_lnrf

    assert allclose_recon, (
        f"LNRF tetrad reconstruction of inverse metric failed for {st.__class__.__name__} with dtype {dtype} on {device}\n"
        f"Errors: MIN: {err_recon.abs().min():.2e}, MAX: {err_recon.abs().max():.2e}, TOTAL: {err_recon.norm():.2e} "
    )

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("x_uncast", x_coords_uncast)
def test_schwarzschild_equivalence(x_uncast: torch.Tensor, dtype: torch.dtype, device: str):
    x = x_uncast.to(dtype=dtype, device=device)
    st_ss = SphericallySymmetric()
    st_kerr0 = KerrBL(a=0.0)
    st_kn00 = KerrNewmanBL(a=0.0, q=0.0)

    g_ss = st_ss.g(x)
    g_kerr0 = st_kerr0.g(x)
    g_kn00 = st_kn00.g(x)

    tol = 1e-6 if dtype == torch.float64 else 1e-4
    assert torch.allclose(g_ss, g_kerr0, atol=tol, rtol=tol)
    assert torch.allclose(g_ss, g_kn00, atol=tol, rtol=tol)
