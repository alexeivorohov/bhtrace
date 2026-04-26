import torch
import pytest
from bhtrace.utils.functional import WeierstrassElliptic

def test_weierstrass_properties():
    # Lemniscatic case: g2=1, g3=0
    g2_val = 1.0
    g3_val = 0.0
    weierstrass_func = WeierstrassElliptic(g2_val, g3_val)

    z = torch.tensor([0.5, 1.0, 1.5])
    
    # Test even property of p(z)
    p_pos = weierstrass_func(z)
    p_neg = weierstrass_func(-z)
    assert torch.allclose(p_pos, p_neg, atol=1e-4), "p(z) should be an even function"

    # Test odd property of p'(z)
    dp_pos = weierstrass_func.d(z)
    dp_neg = weierstrass_func.d(-z)
    assert torch.allclose(dp_pos, -dp_neg, atol=1e-4), "p'(z) should be an odd function"

    # Test if the differential equation holds: (p')^2 = 4p^3 - g2*p - g3
    p_vals = weierstrass_func(z)
    dp_vals = weierstrass_func.d(z)
    
    lhs = dp_vals**2
    rhs = 4 * p_vals**3 - g2_val * p_vals - g3_val
    
    assert torch.allclose(lhs, rhs, atol=1e-3), "The differential equation is not satisfied"

def test_weierstrass_zero():
    g2_val = 1.0
    g3_val = 0.0
    weierstrass_func = WeierstrassElliptic(g2_val, g3_val)

    z_zero = torch.tensor(0.0)
    p_zero = weierstrass_func(z_zero)
    dp_zero = weierstrass_func.d(z_zero)

    assert torch.isinf(p_zero), "p(0) should be infinity"
    assert torch.isinf(dp_zero), "p'(0) should be infinity"
