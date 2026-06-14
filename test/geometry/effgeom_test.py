import torch
import pytest


import bhtrace.geometry.spacetime as st
import bhtrace.physics.electrodynamics as bhE
import bhtrace.utils.units as bhU

from bhtrace.utils.routines import xz_grid_4d_spherical

base = st.MinkowskiSph()

units = bhU.NaturalGeometric(3.0)
B0 = (bhU.schwinger_B / 100.0).to(units).f

field_A = bhE.SplitMonopole(B0)

model_A = bhE.Maxwell(units)
model_B = bhE.EulerHeisenberg(units)
# model_C = bhE.BornInfeld()
# model_D = bhE.


EFF_SPACETIMES = [
    st.EffectiveGeometry(base, model_A, field_A),
    st.EffectiveGeometry(base, model_B, field_A),
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

@pytest.mark.parametrize("spacetime", EFF_SPACETIMES)
@pytest.mark.parametrize("x", COORDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", NUMERIC_TOLERANCES)
class TestSpacetime:

    def test_metric(
        self,
        spacetime: st.Spacetime,
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
            f"Identity check failed - "
            f"Errors: MIN: {err.abs().min():.2e}, MAX: {err.abs().max():.2e}, TOTAL: {err.norm():.2e} "
        )
