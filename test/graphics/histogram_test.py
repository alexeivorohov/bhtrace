import pytest
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import List

np.random.seed(42)

import bhtrace.graphics as bhg
from bhtrace.graphics.histogram import _process_ridge_data

OUTPUT_DIR = 'test/outputs'

def _get_filename(request):
    """Generate a sanitized filename from the pytest request object."""
    module_name = os.path.basename(request.fspath).replace('_test.py', '')
    
    test_case_name = request.node.name
    test_case_name = re.sub(r'\[', '_', test_case_name)
    test_case_name = re.sub(r'\]', '', test_case_name)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', test_case_name)
    
    return f"{module_name}_{sanitized_name}.png"

# --- Parameters ---
SHAPES = [(128,), (16, 64), (4, 16, 16)]
DATA_GENERATORS = {
    "gaussian": lambda shape: np.random.randn(*shape),
    "lognormal": lambda shape: np.random.lognormal(size=shape)
}

# --- Fixtures ---
@pytest.fixture(scope="session", autouse=True)
def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

@pytest.fixture
def data_for_ridge() -> List[np.ndarray]:
    return [np.random.randn(n) for n in np.random.randint(50, 150, size=10)]

@pytest.fixture
def weights_for_ridge(data_for_ridge: List[np.ndarray]) -> List[np.ndarray]:
    return [np.random.rand(*d.shape) for d in data_for_ridge]


# --- Helper functions ---
def save_mpl_figure(fig, request):
    filename = _get_filename(request)
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)

# --- Tests ---

def test_process_ridge_data(data_for_ridge, weights_for_ridge):
    bins = np.linspace(-3, 3, 11)
    
    # Test with weights
    dist = _process_ridge_data(data_for_ridge, bins, weights=weights_for_ridge, density=True)
    assert isinstance(dist, list)
    assert len(dist) == len(data_for_ridge)
    assert all(isinstance(d, np.ndarray) for d in dist)
    assert all(d.shape == (len(bins) - 1,) for d in dist)

    # Test without weights
    dist_no_weights = _process_ridge_data(data_for_ridge, bins, weights=None, density=False)
    assert len(dist_no_weights) == len(data_for_ridge)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("gen_name", DATA_GENERATORS.keys())
@pytest.mark.parametrize("backend", ["mpl", "uniplot"])
def test_histogram(shape, gen_name, backend, request):
    gen_func = DATA_GENERATORS[gen_name]
    data = gen_func(shape)
    
    fig, ax = bhg.hist(data, backend=backend, label=f"{gen_name} data")
    
    if backend == "mpl":
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        save_mpl_figure(fig, request)
    else:
        assert fig is None
        assert ax is None

@pytest.mark.parametrize("backend", ["mpl", "uniplot"])
@pytest.mark.parametrize("bin_scale", ["linear", "log"])
def test_ridge(data_for_ridge, backend, bin_scale, request):
    if bin_scale == "log":
        # Use lognormal data for log scale test
        data = [np.random.lognormal(size=d.shape) for d in data_for_ridge]
    else:
        data = data_for_ridge
        
    fig, ax = bhg.ridge(data, backend=backend, bin_scale=bin_scale, bins=25)
    
    if backend == "mpl":
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.name == '3d'
        save_mpl_figure(fig, request)
    else:
        assert fig is None
        assert ax is None

def test_histogram_mpl_specifics(request):
    data = np.random.randn(100)
    bins = np.array([-3, -1, 0, 1, 3])
    fig, ax = bhg.hist(data, bins=bins, p_scale='log', density=False)
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_yscale() == 'log'
    save_mpl_figure(fig, request)

def test_ridge_mpl_specifics(data_for_ridge, request):
    parameter = np.linspace(0, 1, len(data_for_ridge))
    fig, ax = bhg.ridge(data_for_ridge, parameter=parameter, scale='log', bins=15)
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_zscale() == 'log'
    save_mpl_figure(fig, request)
