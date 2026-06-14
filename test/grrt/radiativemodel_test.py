import pytest
import torch

import bhtrace.geometry.spacetime as st
from bhtrace.grrt.radiation import RadiativeModel, Blackbody, BolometricFlux
import bhtrace.medium as bhM
from bhtrace.utils.routines import xz_grid_4d_spherical



MODELS = [
    Blackbody(torch.logspace(1, 12, 64)),
    BolometricFlux(),
]

MEDIUMS = [
    bhM.VolumetricShell(spacetime=st.KerrBL(), mass=3.0),
    bhM.KeplerianDisk(spacetime=st.KerrBL(), mass=3.0),
]


N_EVAL = 4
COORDS = [
    xz_grid_4d_spherical(N_EVAL, N_EVAL, (6.0, 15.0))
]

DEVICES = [
    'cpu',
    'cuda',
]

DTYPES = [
    # torch.float32,
    torch.float64,
]

ZERO_TOL = 1e-4

@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('medium', MEDIUMS)
@pytest.mark.parametrize('x', COORDS)
@pytest.mark.parametrize('intensity_scale', [1e-2, 1.0, 1e2])
@pytest.mark.parametrize('z_scale', [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('lmbda_scale', [1e-2, 1.0])
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('trace', [True, False])
def test_radiative_model_step(
    model: RadiativeModel, 
    medium: bhM.Medium, 
    x: torch.Tensor, 
    intensity_scale: float,
    z_scale: float,
    lmbda_scale: float,
    device: str, 
    dtype: torch.dtype,
    trace: bool
):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = x.to(device=device, dtype=dtype)

    dlambda = torch.full(x.shape[:-1], lmbda_scale, device=device, dtype=dtype)
    z = torch.ones_like(x[..., 0])*z_scale
    intensity = model.init(batch_shape=x[..., 0].shape, device=device, dtype=dtype, trace=trace).fill_(intensity_scale)

    is_zero_expected = not (intensity_scale > 0)
    try:
        new_intensity = model.step(x=x, intensity=intensity, z=z, dlambda=dlambda, medium=medium)
    except Exception as e:
        pytest.fail(f"model.step failed with {e} on device {device} with dtype {dtype}")

    assert isinstance(new_intensity, torch.Tensor), (
        "`model.step() should return torch.Tensor"
        f"got {type(new_intensity)}"
    )
    assert not new_intensity.isnan().any(), "NaNs in output"
    assert not new_intensity.isinf().any(), "Infs in output"
    assert not new_intensity.less(0.0).any(), "Negative intensities"
    assert intensity.shape == new_intensity.shape, "Output shape mismatch"

    if is_zero_expected:
        assert new_intensity.abs().less(ZERO_TOL).all(), "Got non-zero inteisities when zero intensities expected"
    else:
        assert new_intensity.greater(0.0).any(), "Got zero intensities when non-zero intensities expected"
