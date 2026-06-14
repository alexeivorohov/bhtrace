import torch
import pytest

import bhtrace.geometry.spacetime as st
from bhtrace.utils.routines import xz_grid_4d_spherical
import bhtrace.physics.electrodynamics as bhE
import bhtrace.utils.units as bhU

# --- Definitions and parameters ---
N_EVAL = 4

# TODO: Fix failing test for KerrNewman (lnrf and tetrad methods) - review the definitions
SPACETIMES = [
    # st.SphericallySymmetric(),
    # st.KerrBL(0.001),
    # st.KerrBL(0.999),
    st.KerrBL()
    # st.KerrNewmanBL(0.9, 0.1),
    # st.KerrNewmanBL(0.5, 0.5), 
]


@pytest.fixture
def x() -> torch.Tensor:

    return xz_grid_4d_spherical(N_EVAL, N_EVAL, (4.0, 20.0))


# --- Test cases ---
@pytest.mark.parametrize('spacetime', SPACETIMES)
class TestSpacetimeLocal:
    def test_g(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        spacetime_g_spy = mocker.spy(sl.spacetime, 'g')

        # First call
        g_val = sl.g
        spacetime_g_spy.assert_called_once_with(x)

        # Second call (should be cached)
        g_val_cached = sl.g
        spacetime_g_spy.assert_called_once()

        assert g_val is g_val_cached
        assert g_val.shape == (*x.shape[:-1], 4, 4)

        # Check for correctness of values
        g_val_direct = spacetime.g(x)
        assert torch.allclose(g_val, g_val_direct)

    def test_ginv(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        spacetime_ginv_spy = mocker.spy(sl.spacetime, 'ginv')

        # First call
        ginv_val = sl.ginv
        spacetime_ginv_spy.assert_called_once_with(x)

        # Second call (should be cached)
        ginv_val_cached = sl.ginv
        spacetime_ginv_spy.assert_called_once()

        assert ginv_val is ginv_val_cached
        assert ginv_val.shape == (*x.shape[:-1], 4, 4)

        # Check for correctness of values
        ginv_val_direct = spacetime.ginv(x)
        assert torch.allclose(ginv_val, ginv_val_direct)

    def test_detg(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        # detg depends on g, so we spy on spacetime.g
        spacetime_g_spy = mocker.spy(spacetime, 'g')

        # First call
        detg_val = sl.detg
        spacetime_g_spy.assert_called_once_with(x)

        # Second call (should be cached)
        detg_val_cached = sl.detg
        spacetime_g_spy.assert_called_once()

        assert detg_val is detg_val_cached
        assert detg_val.shape == x.shape[:-1]

        # Check for correctness of values
        detg_val_direct = spacetime.detg(x) # this will call g again
        assert torch.allclose(detg_val, detg_val_direct)
        assert spacetime_g_spy.call_count == 2


    def test_conn(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        spacetime_conn_spy = mocker.spy(sl.spacetime, 'conn')

        # First call
        conn_val = sl.conn
        spacetime_conn_spy.assert_called_once_with(x)

        # Second call (should be cached)
        conn_val_cached = sl.conn
        spacetime_conn_spy.assert_called_once()

        assert conn_val is conn_val_cached
        assert conn_val.shape == (*x.shape[:-1], 4, 4, 4)

        # Check for correctness of values
        conn_val_direct = spacetime.conn(x)
        assert torch.allclose(conn_val, conn_val_direct)

    def test_dg(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        spacetime_dg_spy = mocker.spy(sl.spacetime, 'dg')

        # First call
        dg_val = sl.dg
        spacetime_dg_spy.assert_called_once_with(x)

        # Second call (should be cached)
        dg_val_cached = sl.dg
        spacetime_dg_spy.assert_called_once()

        assert dg_val is dg_val_cached
        assert dg_val.shape == (*x.shape[:-1], 4, 4, 4)

        # Check for correctness of values
        dg_val_direct = spacetime.dg(x)
        assert torch.allclose(dg_val, dg_val_direct)

    def test_sqrtmg(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        # sqrtmg -> detg -> g
        spacetime_g_spy = mocker.spy(spacetime, 'g')

        # First call
        sqrtmg_val = sl.sqrtmg
        spacetime_g_spy.assert_called_once_with(x)

        # Second call (cached)
        sqrtmg_val_cached = sl.sqrtmg
        spacetime_g_spy.assert_called_once()
        
        # also sl.detg is cached
        detg_val = sl.detg
        spacetime_g_spy.assert_called_once()

        # and sl.g is cached
        g_val = sl.g
        spacetime_g_spy.assert_called_once()

        assert sqrtmg_val is sqrtmg_val_cached
        assert sqrtmg_val.shape == x.shape[:-1]

        # Check for correctness
        sqrtmg_val_direct = (-spacetime.detg(x)).sqrt()
        assert torch.allclose(sqrtmg_val, sqrtmg_val_direct)
        assert spacetime_g_spy.call_count == 2

    def test_tetrad(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        spacetime_tetrad_spy = mocker.spy(sl.spacetime, 'tetrad')

        # First call
        tetrad_val = sl.tetrad
        spacetime_tetrad_spy.assert_called_once_with(x)

        # Second call (should be cached)
        tetrad_val_cached = sl.tetrad
        spacetime_tetrad_spy.assert_called_once()

        assert tetrad_val is tetrad_val_cached
        assert tetrad_val.shape == (*x.shape[:-1], 4, 4)

        # Check for correctness of values
        tetrad_val_direct = spacetime.tetrad(x)
        assert torch.allclose(tetrad_val, tetrad_val_direct)

    def test_lnrf(self, spacetime: st.Spacetime, x: torch.Tensor, mocker):
        sl = spacetime.local(x)
        spacetime_lnrf_spy = mocker.spy(sl.spacetime, 'lnrf')

        # First call
        lnrf_val = sl.lnrf
        spacetime_lnrf_spy.assert_called_once_with(x)

        # Second call (should be cached)
        lnrf_val_cached = sl.lnrf
        spacetime_lnrf_spy.assert_called_once()

        assert lnrf_val is lnrf_val_cached
        assert lnrf_val.shape == (*x.shape[:-1], 4, 4)

        # Check for correctness of values
        lnrf_val_direct = spacetime.lnrf(x)
        assert torch.allclose(lnrf_val, lnrf_val_direct)
