from typing import Tuple

import pytest
import torch
from scipy import constants
import inspect

import bhtrace.geometry.spacetime as ST
from bhtrace.grrt.radiation import RadiativeModel, Blackbody, IntegralFlux
from bhtrace.medium._mock import MockMedium
from bhtrace.medium._base import Medium

def fetch_coords(n_eval=10, r_range=(10, 20.0)):
    # Avoid coordinate singularities and event horizon
    coords = torch.rand(n_eval, 4)
    coords[:, 1] = coords[:, 1] * (r_range[1] - r_range[0]) + r_range[0]  # r
    coords[:, 2] = (
        coords[:, 2] * (0.9 * torch.pi) + 0.05 * torch.pi
    )  # theta, away from 0 and pi
    coords[:, 3] = coords[:, 3] * 2 * torch.pi  # phi
    return coords


MODELS = [
    Blackbody(),
    IntegralFlux(),
]

MEDIUMS = [
    MockMedium(),
    MockMedium(spacetime=ST.MinkowskiSph()),
]


N_EVAL = 4
COORDS = [
    fetch_coords(N_EVAL, (10, 20)),
    fetch_coords(N_EVAL*2, (6, 12)),
]

LMBDA_SCALE = [
    (0.1, 0.01)
]

N_FREQS = 32
FREQS = [
    torch.logspace(1, 5, N_FREQS),
]

DEVICES = [
    'cpu',
    'cuda:0',
]

DTYPES = [
    torch.float32,
    torch.float64,
]

@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('medium', MEDIUMS)
@pytest.mark.parametrize('x', COORDS)
@pytest.mark.parametrize('lmbda_scale', LMBDA_SCALE)
@pytest.mark.parametrize('nu', FREQS)
@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_radiative_model(model: RadiativeModel, medium: Medium, x: torch.Tensor, nu: torch.Tensor, lmbda_scale: Tuple[float], device: str, dtype: torch.dtype):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nu = nu.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    medium._r = medium._r.to(device=device, dtype=dtype)
    medium._temp = medium._temp.to(device=device, dtype=dtype)
    medium._density = medium._density.to(device=device, dtype=dtype)
    medium._opacity = medium._opacity.to(device=device, dtype=dtype)
    medium._flux = medium._flux.to(device=device, dtype=dtype)


    if model.spectral:
        inv_i_prev = torch.ones(x.shape[0], nu.shape[0], device=device, dtype=dtype)
        nu_comoving = nu.expand(x.shape[0], -1)
    else:
        inv_i_prev = torch.ones(x.shape[0], device=device, dtype=dtype)
        nu_comoving = None # Not used for integral flux

    dlambda = torch.full(x.shape[:-1], lmbda_scale[0], device=device, dtype=dtype)
    z = torch.ones_like(x[..., 0])

    step_params = inspect.signature(model.step).parameters
    kwargs = {'inv_i_prev': inv_i_prev, 'medium': medium, 'x': x}
    if 'nu_comoving' in step_params and nu_comoving is not None:
        kwargs['nu_comoving'] = nu_comoving
    if 'dlambda' in step_params:
        kwargs['dlambda'] = dlambda
    if 'z' in step_params:
        kwargs['z'] = z

    try:
        new_inv_i = model.step(**kwargs)
    except Exception as e:
        pytest.fail(f"model.step failed with {e} on device {device} with dtype {dtype}")

    assert new_inv_i is not None
    assert not torch.isnan(new_inv_i).any(), "NaNs in output"
    assert not torch.isinf(new_inv_i).any(), "Infs in output"
    assert new_inv_i.shape == inv_i_prev.shape, "Output shape mismatch"

