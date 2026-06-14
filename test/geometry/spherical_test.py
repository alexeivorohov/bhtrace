import torch
import pytest

from bhtrace.geometry import Spacetime
import bhtrace.geometry.spacetime.spherical as sph

from bhtrace.utils.routines import xz_grid_4d_spherical

# Collection of spacetimes to test
SPACETIMES = [
    sph.MinkowskiSph(),
    sph.SphericallySymmetric(),  # Schwarzschild
    sph.SphericallySymmetric(A=lambda r: 1 - 2 / r + r**-2, A_r=lambda r: 2 * r**-2 - 2 * r**-3),  # Reissner-Nordstrom with M=Q=1
    sph.KerrBL(a=0.0),
    sph.KerrBL(a=0.9),
    sph.KerrNewmanBL(a=0.8, q=0.3),
]


N_EVAL = 4
COORDS = [
    xz_grid_4d_spherical(N_EVAL, N_EVAL, (3.0, 5.0), (-3, -3)), # near
    xz_grid_4d_spherical(N_EVAL, N_EVAL, (5.0, 10.0), (-5.0, 5.0)), # mediate
    xz_grid_4d_spherical(N_EVAL, N_EVAL, (10.0, 20.0), (-10.0, 10.0)), # far
    xz_grid_4d_spherical(N_EVAL, N_EVAL, (-0.5, 0.5), (3.0, 10.0)), # pole
]

NUMERIC_TOLERANCES = [1e-1, 1e-4]

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

DTYPES = [
    torch.float32,
    torch.float64,
]

@pytest.mark.parametrize("spacetime", SPACETIMES)
@pytest.mark.parametrize("x", COORDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", NUMERIC_TOLERANCES)
class TestSpacetime:

    def test_metric(
        self,
        spacetime: Spacetime,
        x: torch.Tensor,
        dtype: torch.dtype,
        device: str,
        eps: float,
    ):
        """Test if g @ ginv = I."""
        x = x.to(dtype=dtype, device=device)

        g = spacetime.g(x)

        assert g.dtype == dtype, (
            f"`g` dtype {g.dtype} not matches test dtype {dtype}"
        )

        assert str(g.device) == device, (
            f"`g` device {g.device} not matches test device {device}"
        )
        
        ginv = spacetime.ginv(x)
        assert  ginv.dtype == dtype, (
           f"`ginv` dtype {ginv.dtype} not matches test dtype {dtype}"
        )
        
        assert str(ginv.device) == device, (
            f"`ginv` device {ginv.device} not matches test device {device}"
        )
        
        identity = torch.eye(4, dtype=x.dtype, device=x.device)
        gginv = torch.einsum('...uv, ...vq -> ...uq', g, ginv)

        allclose = torch.allclose(gginv, identity.expand_as(gginv), atol=eps, rtol=eps)

        err = gginv - identity.expand_as(gginv)

        assert allclose, (
            f"Identity check failed for {spacetime.__class__.__name__}"
            f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e} "
        )

    def test_conn_vs_numerical(
        self,
        spacetime: Spacetime,
        x: torch.Tensor,
        dtype: torch.dtype,
        device: str,
        eps: float,
    ):
        """Test if analytic connection matches numerical."""

        if not (
            hasattr(spacetime, "__analytic_conn__") and spacetime._has_analytic_conn
        ):
            pytest.skip(
                f"{spacetime.__class__.__name__} does not have an analytic connection."
            )

        if dtype == torch.float32 and eps < 1e-2:
            pytest.skip("High precision test for float32 is unstable")

        x = x.to(dtype=dtype, device=device)
        spacetime._eps = eps

        conn_analytic = spacetime.conn(x)
        conn_numerical = spacetime.conn_(x)

        tol = 10 * spacetime._eps
        allclose = torch.allclose(conn_analytic, conn_numerical, atol=tol, rtol=tol)

        err = conn_analytic - conn_numerical

        assert allclose, (
            f"Connection vs numerical failed for {spacetime.__class__.__name__} with dtype {dtype} on {device} and eps {eps:.1e}\n"
            f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e} "
        )

    def test_analytic_tetrad(
        self,
        spacetime: Spacetime,
        x: torch.Tensor,
        dtype: torch.dtype,
        device: str,
        eps: float,
    ):
        """Test if tetrad reconstructs the inverse metric."""
        x = x.to(dtype=dtype, device=device)
        try:
            e_i_mu = spacetime.tetrad(x)
        except (NotImplementedError, AttributeError):
            pytest.skip(f"Tetrad not implemented for {spacetime.__class__.__name__}")
            return
        
        assert e_i_mu is not None, (
            f"Tetrad returned none for {spacetime.__class__.__name__}"
        )

        nans = e_i_mu.isnan()
        assert not torch.any(nans), (
            f"Tetrad returned nans for samples {nans.any(dim=-1).any(dim=-1)}"
        )

        eta = torch.diag(
            torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=x.dtype, device=x.device)
        )
        ginv = spacetime.ginv(x)

        ginv_recon = torch.einsum("...ia, ab,...jb->...ij", e_i_mu, eta, e_i_mu)

        allclose = torch.allclose(ginv, ginv_recon, atol=eps * 10, rtol=eps * 10)

        err = ginv - ginv_recon

        assert allclose, (
            # f"Tetrad reconstruction of inverse metric failed for {spacetime.__class__.__name__} with dtype {dtype} on {device}\n"
            f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e} "
        )

    def test_taylor(
        self,
        spacetime: Spacetime,
        x: torch.Tensor,
        dtype: torch.dtype,
        device: str,
        eps: float,
    ):
        """Checks if metric at a nearby point can be approximated by a Taylor expansion using `dg`."""
        if dtype == torch.float32 and eps < 1e-4:
            pytest.skip("High precision test for float32 is unstable")

        x = x.to(dtype=dtype, device=device)
        spacetime._eps = 1e-5 if dtype == torch.float64 else 1e-3

        dx = torch.randn_like(x)
        dx = dx / torch.linalg.norm(dx, dim=-1, keepdim=True) * eps

        g_x = spacetime.g(x)
        dg_x = spacetime.dg(x)
        g_x_plus_dx = spacetime.g(x + dx)

        g_taylor = g_x + torch.einsum("...puv,...p->...uv", dg_x, dx)

        tol = 100 * eps**2
        allclose = torch.allclose(g_x_plus_dx, g_taylor, atol=tol, rtol=tol)

        err = g_x_plus_dx - g_taylor

        assert allclose, (
            # f"Taylor expansion failed for {spacetime.__class__.__name__} with eps={eps:.1e}, dtype {dtype} on {device}\n"
            f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e}"
        )

    def test_lnrf(
        self,
        spacetime: Spacetime,
        x: torch.Tensor,
        dtype: torch.dtype,
        device: str,
        eps: float,
    ):
        """Test if LNRF tetrad can be computed and reconstructs the inverse metric."""
        x = x.to(dtype=dtype, device=device)
        try:
            e_i_mu_lnrf = spacetime.lnrf(x)
        except (NotImplementedError, AttributeError):
            pytest.skip(f"LNRF not implemented for {spacetime.__class__.__name__}")
            return

        assert e_i_mu_lnrf is not None, (
            f"LNRF tetrad returned none for {spacetime.__class__.__name__}"
        )

        nans = e_i_mu_lnrf.isnan()
        assert not torch.any(nans), (
            f"LNRF tetrad returned nans for samples {nans.any(dim=-1).any(dim=-1)}"
        )

        spacetime._eps = eps
        eta = torch.diag(
            torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=x.dtype, device=x.device)
        )
        ginv = spacetime.ginv(x)

        ginv_recon_lnrf = torch.einsum(
            "...ia, ab,...jb->...ij", e_i_mu_lnrf, eta, e_i_mu_lnrf
        )

        tol = 10 * spacetime._eps
        allclose_recon = torch.allclose(ginv, ginv_recon_lnrf, atol=tol, rtol=tol)

        err_recon = ginv - ginv_recon_lnrf

        assert allclose_recon, (
            # f"LNRF tetrad reconstruction of inverse metric failed for {spacetime.__class__.__name__} with dtype {dtype} on {device}\n"
            f"Errors: MIN: {err_recon.abs().min():.2e}, MAX: {err_recon.abs().max():.2e}, TOTAL: {err_recon.norm():.2e} "
        )
