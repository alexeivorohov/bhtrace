from typing import Optional, List, Dict, Callable, Any

import torch
import pytest

from bhtrace.utils.diff import jacobian

# --- vector functions ---


def const_vector(x: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
    a = a or torch.ones(x.shape, dtype=x.dtype, device=x.device)
    return a


def diff_const_vector(
    x: torch.Tensor, a: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.zeros(x.shape, dtype=x.dtype, device=x.device)


def poly_vector(
    x: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    powers: Optional[List[int]] = None,
) -> torch.Tensor:
    a = a or torch.eye(x.shape[:-1], dtype=x.dtype, device=x.device)
    powers = powers or [1]
    x_ = x @ a

    outp = [x_.pow(i) for i in powers]

    return sum(outp)


def diff_poly_vector(
    x: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    powers: Optional[List[int]] = None,
):
    a = a or torch.eye(x.shape[:-1], dtype=x.dtype, device=x.device)
    powers = powers or [1]
    x_ = x.bmm(a)

    outp = [i * torch.bmm(a, x_.pow(i - 1)) for i in powers]
    return sum(outp)


def exp_vector(x: torch.Tensor, a: Optional[torch.Tensor] = None):
    a = a or torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    return torch.exp(x.bmm(a))


def diff_exp_vector(x: torch.Tensor, a: Optional[torch.Tensor] = None):
    a = a or torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    return torch.bmm(a, torch.exp(x.bmm(a)))


# --- hamiltonian ---


def hamiltonian(
    x: torch.Tensor, p: torch.Tensor, a: Optional[torch.Tensor] = None
) -> torch.Tensor:
    a = a or torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device).unsqueeze(-1)
    return p.pow(2) + a * x.pow(2)


def diff_hamiltonian(
    x: torch.Tensor,
    p: torch.Tensor,
    a: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a = a or torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device).unsqueeze(-1)
    return 2 * a * x


# --- fixtures ---


@pytest.fixture
def f_dict():
    """Test functions and their expected derivatives."""
    return {
        "const_vec": const_vector,
        "poly_vec": poly_vector,
        "exp_vec": exp_vector,
        "hmlt": hamiltonian,
    }


@pytest.fixture
def diff_dict():
    """Expected derivatives for test functions."""
    return {
        "const_vec": diff_const_vector,
        "poly_vec": diff_poly_vector,
        "exp_vec": diff_exp_vector,
        "hmlt": hamiltonian,
    }


@pytest.fixture
def param_dict() -> Dict[str, List[Dict[str, Any]]]:
    """Parameters for test functions"""
    return {
        "const": [{"args": None, "kwargs": None}, {...}],
        "hmlt": [
            {"args": None, "kwargs": None},
        ],
    }


# --- test run ---

def _test_jacobian(
    func_name: str,
    func: Callable[[Any], torch.Tensor],
    diff_func: Callable[[Any], torch.Tensor],
    x: torch.Tensor,
    args: List,
    kwargs: Dict,
    order: int,
    eps: float,
    dtype: torch.dtype,
    device: torch.device,
):
    """Test Jacobian computation"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = x.to(device, dtype)
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg = arg.to(device, dtype)
    for arg in kwargs.values():
        if isinstance(arg, torch.Tensor):
            arg = arg.to(device, dtype)

    result = jacobian(func, x, eps=eps, order=order, args=args, kwargs=kwargs)
    expected = diff_func(x, *args, **kwargs)

    dlta = torch.abs(result - expected)
    tol_condition = torch.allclose(result, expected, atol=eps * 10, rtol=eps * 10)

    assert tol_condition, (
        f"Given tolerance {eps} is not achieved: MAE={dlta.mean():.2e}, MIN={dlta.min():.2e}, MAX={dlta.max():.2e}\n"
        f"Result:\n\t{result}\nExpected:\n\t{expected}\n"
        f"Function: {func_name}, scheme order: {order}"
    )


@pytest.mark.parametrize("func_name", ["const_vec", "poly_vec", "exp_vec", "hmlt"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("input_shape", [[1, 1, 4], [16, 4], [1, 32, 2, 4]])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("eps", [1e1, 1e-2, 1e-4])
def test_jacobian(
    f_dict,
    diff_dict,
    param_dict,
    func_name,
    order,
    eps,
    dtype,
    device,
    input_shape,
):
    """Test Jacobian computation"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.randn(input_shape).to(dtype=dtype, device=device) * 2.0

    func = f_dict[func_name]
    diff_func = diff_dict[func_name]
    params = param_dict.get(func, {})

    for param in params:
        args = param.get("args", [])
        kwargs = param.get("kwargs", {})
        if func_name == "hmlt":
            args = [torch.randn_like(x)].extend()  # 'velocities'
        _test_jacobian(
            func_name, func, diff_func, x, args, kwargs, order, eps, dtype, device
        )


@pytest.mark.parametrize("order", [5, 6])
def test_jacobian_invalid_order(order):
    """Test that invalid orders raise ValueError."""
    if order not in [1, 2, 4]:
        X = torch.randn(2, 3).to(dtype=torch.float64)
        f = lambda X: X

        with pytest.raises(KeyError):
            jacobian(f, X, order=order)
