import torch
import pytest

import bhtrace.physics.electrodynamics as bhE
import bhtrace.utils.units as bhU
import bhtrace.geometry.spacetime as st
from bhtrace.utils.routines import xz_grid_4d_spherical
from bhtrace.physics.electrodynamics.fields._base import ElectromagneticLocal

SPACETIMES = [
    # st.MinkowskiCart(),
    st.KerrBL(),
]

COORDS = [
    xz_grid_4d_spherical(1, 1, x_lims=(4.0, 20.0))
]

MODELS = [
    bhE.Maxwell(bhU.si),
    bhE.EulerHeisenberg(bhU.si),
]

FIELDS = [
    bhE.PointCharge(1.0, 0.0),
    bhE.PointCharge(0.0, 1.0),
]

@pytest.mark.parametrize('spacetime', SPACETIMES)
@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('x', COORDS)
class TestElectrodynamicsLocal:
    def test_E(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        direct = field.E(x)
        f_spy = mocker.spy(f_loc.field, 'E')

        # First call
        val1 = f_loc.E
        f_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.E
        f_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1], 4)
        assert direct.shape == val1.shape
        assert torch.allclose(direct, val1)

    def test_B(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        direct = field.B(x)
        f_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.B
        f_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.B
        f_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1], 4)
        assert direct.shape == val1.shape
        assert torch.allclose(direct, val1)

    def test_U2(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        expected = - torch.ones_like(u[..., 0])

        u2 = torch.einsum('...a, ...a -> ...', f_loc.u, f_loc.U_d)
        direct_u2 = torch.einsum("...u, ...uv, ...v -> ...", f_loc.u, st_loc.g, f_loc.u)

        assert direct_u2.shape == expected.shape
        assert direct_u2.allclose(expected)

        assert u2.shape == expected.shape
        assert u2.allclose(expected)

        assert torch.allclose(direct_u2, u2)

    def test_E2(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        f_spy = mocker.spy(f_loc.field, 'E')

        # First call
        val1 = f_loc.E2
        f_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.E2
        f_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1],)

        direct_e = field.E(x)
        direct_e2 = torch.einsum("...u, ...uv, ...v -> ...", direct_e, st_loc.g, direct_e)
        assert direct_e2.shape == val1.shape
        assert torch.allclose(direct_e2, val1)

    def test_B2(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        f_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.B2
        f_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.B2
        f_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1],)

        direct_b = field.B(x)
        direct_b2 = torch.einsum("...u, ...uv, ...v -> ...", direct_b, st_loc.g, direct_b)
        assert direct_b2.shape == val1.shape
        assert torch.allclose(direct_b2, val1)

    def test_F_invariant(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        e_spy = mocker.spy(f_loc.field, 'E')
        b_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.F
        e_spy.assert_called_once_with(x)
        b_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.F
        e_spy.assert_called_once()
        b_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1],)

        direct_e = field.E(x)
        direct_b = field.B(x)
        direct_e2 = torch.einsum("...u, ...uv, ...v -> ...", direct_e, st_loc.g, direct_e)
        direct_b2 = torch.einsum("...u, ...uv, ...v -> ...", direct_b, st_loc.g, direct_b)
        direct_f = 2 * (direct_b2 - direct_e2)
        assert direct_f.shape == val1.shape
        assert torch.allclose(direct_f, val1)

    def test_G_invariant(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        e_spy = mocker.spy(f_loc.field, 'E')
        b_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.G
        e_spy.assert_called_once_with(x)
        b_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.G
        e_spy.assert_called_once()
        b_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1],)

        direct_e = field.E(x)
        direct_b = field.B(x)
        direct_g = 4 * torch.einsum("...u, ...uv, ...v -> ...", direct_e, st_loc.g, direct_b)
        assert direct_g.shape == val1.shape
        assert torch.allclose(direct_g, val1)

    def test_L(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        e_spy = mocker.spy(f_loc.field, 'E')
        b_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.L
        e_spy.assert_called_once_with(x)
        b_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.L
        e_spy.assert_called_once()
        b_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1],)

        direct_e = field.E(x)
        direct_b = field.B(x)
        direct_e2 = torch.einsum("...u, ...uv, ...v -> ...", direct_e, st_loc.g, direct_e)
        direct_b2 = torch.einsum("...u, ...uv, ...v -> ...", direct_b, st_loc.g, direct_b)
        direct_f = 2 * (direct_b2 - direct_e2)
        direct_g_inv = 4 * torch.einsum("...u, ...uv, ...v -> ...", direct_e, st_loc.g, direct_b)

        direct_l = model.L(direct_f, direct_g_inv)
        assert direct_l.shape == val1.shape
        assert torch.allclose(direct_l, val1)

    def test_L_F(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        e_spy = mocker.spy(f_loc.field, 'E')
        b_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.L_F
        e_spy.assert_called_once_with(x)
        b_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.L_F
        e_spy.assert_called_once()
        b_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1],)

        direct_E = field.E(x)
        direct_B = field.B(x)
        direct_E2 = torch.einsum("...u, ...uv, ...v -> ...", direct_E, st_loc.g, direct_E)
        direct_B2 = torch.einsum("...u, ...uv, ...v -> ...", direct_B, st_loc.g, direct_B)
        direct_F = 2 * (direct_B2 - direct_E2)
        direct_G = 4 * torch.einsum("...u, ...uv, ...v -> ...", direct_E, st_loc.g, direct_B)

        direct_L_F = model.L_F(direct_F, direct_G)
        assert direct_L_F.shape == val1.shape
        assert torch.allclose(direct_L_F, val1)


    def test_F_up(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        e_spy = mocker.spy(f_loc.field, 'E')
        b_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.F_up
        e_spy.assert_called_once_with(x)
        b_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.F_up
        e_spy.assert_called_once()
        b_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1], 4, 4)
        
    def test_FF_up(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        e_spy = mocker.spy(f_loc.field, 'E')
        b_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.FF_up
        e_spy.assert_called_once_with(x)
        b_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.FF_up
        e_spy.assert_called_once()
        b_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1], 4, 4)

    def test_T_dd(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        e_spy = mocker.spy(f_loc.field, 'E')
        b_spy = mocker.spy(f_loc.field, 'B')

        # First call
        val1 = f_loc.T_dd
        e_spy.assert_called_once_with(x)
        b_spy.assert_called_once_with(x)

        # Second call (should be cached)
        val2 = f_loc.T_dd
        e_spy.assert_called_once()
        b_spy.assert_called_once()

        assert val1 is val2
        assert val1.shape == (*x.shape[:-1], 4, 4)

    def test_F_up_antisymmetry(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        F_up = f_loc.F_up
        assert F_up.shape == (-F_up.mT).shape
        assert torch.allclose(F_up, -F_up.mT)

    def test_dual_F_up_antisymmetry(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        dual_F_up = f_loc.dual_F_up
        assert dual_F_up.shape == (-dual_F_up.mT).shape
        assert torch.allclose(dual_F_up, -dual_F_up.mT)

    def test_F_invariant_from_tensors(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        F_up = f_loc.F_up
        F_dd = f_loc.F_dd
        
        F_inv = torch.einsum("...uv, ...uv -> ...", F_up, F_dd)
        assert F_inv.shape == f_loc.F.shape
        assert torch.allclose(F_inv, f_loc.F)

    def test_G_pseudoinvariant_from_tensors(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        dual_F_up = f_loc.dual_F_up
        F_dd = f_loc.F_dd
        
        F_inv = torch.einsum("...uv, ...uv -> ...", dual_F_up, F_dd)
        assert F_inv.shape == f_loc.G.shape
        assert torch.allclose(F_inv, f_loc.G)

    def test_F_ud_trace(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        F_ud = f_loc.F_ud
        trace = torch.einsum("...uu->...", F_ud)
        assert trace.shape == torch.zeros_like(trace).shape
        assert torch.allclose(trace, torch.zeros_like(trace))

    def test_FF_ud_symmetry(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        F_up = f_loc.F_up
        F_ud = f_loc.F_ud
        
        term = torch.einsum("...vx, ...ux -> ...vu", F_up, F_ud)
        assert term.shape == term.mT.shape
        assert torch.allclose(term, term.mT)

    def test_T_up_symmetry(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        T_up = f_loc.T_up
        assert T_up.shape == T_up.mT.shape
        assert torch.allclose(T_up, T_up.mT)

    def test_T_dd_symmetry(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor, 
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)
        
        T_dd = f_loc.T_dd
        assert T_dd.shape == T_dd.mT.shape
        assert torch.allclose(T_dd, T_dd.mT)

    def test_E_from_F(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor,
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        g = f_loc.gX
        u_up = f_loc.u
        u_down = torch.einsum("...uv, ...v -> ...u", g, u_up)

        F_up = f_loc.F_up
        E_from_F = torch.einsum("...uv, ...v -> ...u", F_up, u_down)

        assert E_from_F.shape == f_loc.E.shape
        assert torch.allclose(E_from_F, f_loc.E, atol=1e-5)

    def test_B_from_F_dual(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor,
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        B_from_F_dual = torch.einsum(
            "...uv, ...v -> ...u", f_loc.dual_F_up, f_loc.U_d
        )
        
        assert B_from_F_dual.shape == f_loc.B.shape
        assert torch.allclose(B_from_F_dual, f_loc.B, atol=1e-5)

    def test_eps4_up_value(
        self,
        spacetime: st.Spacetime,
        model: bhE.Electrodynamics,
        field: bhE.ElectromagneticField,
        x: torch.Tensor,
        mocker
    ):
        u = torch.zeros_like(x)
        u[..., 0] = 1.0
        st_loc = spacetime.local(x)
        f_loc = field.local(model, st_loc, u)

        eps4_up = f_loc.eps4_up
        expected_value = 1.0 / st_loc.sqrtmg
        actual_value = eps4_up[..., 0, 1, 2, 3]

        assert actual_value.shape == expected_value.shape
        assert torch.allclose(actual_value, expected_value, atol=1e-5)
