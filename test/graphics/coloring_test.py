from typing import List, Dict

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

from bhtrace.graphics.coloring import (
    _multicolored_lines_2d, _get_scalar_mappable, _multicolored_line_2d_single,
    _normalize_rgba_to_list,
)

# Fixtures for test data
@pytest.fixture
def single_line():
    """A single line."""
    t = np.linspace(0, 1, 50)
    x = np.stack((t, np.sin(t * 5)), axis=-1)
    c = np.linspace(0, 20, 50)
    return x, c

@pytest.fixture
def multiple_lines():
    """Multiple lines of the same length."""
    x_list = []
    c_list = []
    for i in range(3):
        t = np.linspace(0, 1, 50)
        x = np.stack((t, np.cos(t * 3 + i)), axis=-1)
        c = np.linspace(i * 10, i * 10 + 10, 50)
        x_list.append(x)
        c_list.append(c)
    return x_list, c_list

@pytest.fixture
def ragged_lines():
    """Multiple lines of different lengths."""
    x_ragged = []
    c_ragged = []
    
    # Line 1
    t1 = np.linspace(0, 1, 50)
    x1 = np.stack((t1, np.sin(t1 * 5)), axis=-1)
    c1 = np.linspace(0, 20, 50)
    x_ragged.append(x1)
    c_ragged.append(c1)

    # Line 2
    t2 = np.linspace(0, 1, 30)
    x2 = np.stack((t2, np.cos(t2 * 3)), axis=-1) * 0.8
    c2 = np.linspace(10, 30, 30)
    x_ragged.append(x2)
    c_ragged.append(c2)

    return x_ragged, c_ragged


# @pytest.mark.parametrize()
# def test_normalize_rgba_to_list(
#     xy_list : List[np.ndarray],
#     rgba: List[np.ndarray] | np.ndarray,
# ):
    
#     ...


# def test_get_scalar_mappable():
#     c_list = [np.array([1, 2, 3]), np.array([4, 5])]
#     sm = _get_scalar_mappable(c_list, cmap='viridis')
#     assert isinstance(sm, ScalarMappable)
#     assert sm.norm.vmin == 1
#     assert sm.norm.vmax == 5
#     assert sm.cmap.name == 'viridis'

# def test_get_scalar_mappable_with_norm():
#     c_list = [np.array([1, 2, 3]), np.array([4, 5])]
#     norm = Normalize(vmin=0, vmax=10)
#     sm = _get_scalar_mappable(c_list, cmap='viridis', norm=norm)
#     assert sm.norm.vmin == 0
#     assert sm.norm.vmax == 10

# def test_get_scalar_mappable_empty():
#     sm = _get_scalar_mappable([], cmap='viridis')
#     assert isinstance(sm, ScalarMappable)
#     # Default norm values are None
#     assert sm.norm.vmin is None
#     assert sm.norm.vmax is None

# def test_multicolored_line_2d_single(single_line):
#     x, c = single_line
#     fig, ax = plt.subplots()
#     sm = _get_scalar_mappable([c], cmap='plasma')
#     collection = _multicolored_line_2d_single(x, c, ax, linewidth=2, sm=sm)
#     assert isinstance(collection, LineCollection)
#     assert len(collection.get_segments()) == len(x) - 1
#     plt.close(fig)

# def test_multicolored_lines_2d_per_point(multiple_lines):
#     x_list, c_list = multiple_lines
#     fig, ax = plt.subplots()
#     sm = _multicolored_lines_2d(x_list, c_list, cmap='viridis', ax=ax)
#     assert isinstance(sm, ScalarMappable)
#     assert len(ax.collections) == len(x_list)
#     plt.close(fig)

# def test_multicolored_lines_2d_per_line_broadcast(multiple_lines):
#     x_list, _ = multiple_lines
#     c_per_line = np.array([1.0, 2.0, 3.0])
#     fig, ax = plt.subplots()
#     sm = _multicolored_lines_2d(x_list, c_per_line, cmap='magma', ax=ax)
#     assert isinstance(sm, ScalarMappable)
#     assert len(ax.collections) == len(x_list)
#     # Check that each line has a uniform color
#     for i, collection in enumerate(ax.collections):
#         colors = collection.get_colors()
#         # The color should be constant for each segment of a line
#         assert np.all(np.isclose(colors, sm.to_rgba(c_per_line[i])))
#     plt.close(fig)

# def test_multicolored_lines_2d_ragged(ragged_lines):
#     x_ragged, c_ragged = ragged_lines
#     fig, ax = plt.subplots()
#     sm = _multicolored_lines_2d(x_ragged, c_ragged, cmap='cividis', ax=ax)
#     assert isinstance(sm, ScalarMappable)
#     assert len(ax.collections) == len(x_ragged)
#     plt.close(fig)

# def test_multicolored_lines_2d_length_mismatch(multiple_lines):
#     x_list, c_list = multiple_lines
#     c_list.pop() # create a mismatch
#     fig, ax = plt.subplots()
#     with pytest.raises(ValueError, match="For ragged input, lengths of x and c must match."):
#         _multicolored_lines_2d(x_list, c_list, cmap='viridis', ax=ax)
#     plt.close(fig)

# def test_multicolored_lines_2d_invalid_c_shape(multiple_lines):
#     x_list, _ = multiple_lines
#     c_invalid = np.array([[1.0], [2.0], [3.0]]) # Not a 1D array
#     fig, ax = plt.subplots()
#     with pytest.raises(ValueError, match="Invalid shape for c with list x input."):
#         _multicolored_lines_2d(x_list, c_invalid, cmap='viridis', ax=ax)
#     plt.close(fig)

# def test_multicolored_lines_2d_with_cbar(single_line):
#     x, c = single_line
#     fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
#     _multicolored_lines_2d([x], [[c]], cmap='plasma', ax=ax, cbar_ax=cbar_ax)
#     # Check if colorbar has been drawn on cbar_ax
#     assert len(cbar_ax.get_children()) > 1 # More than just background patch
#     plt.close(fig)
