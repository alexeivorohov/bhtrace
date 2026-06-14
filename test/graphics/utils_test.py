import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from bhtrace.graphics.utils import Projector, _normalize_trajectories


# Tests for Projector
@pytest.mark.parametrize("projection_str, in_coords_shape, out_coords_shape_end", [
    ('xy', (10, 3), 2),
    ('yz', (10, 3), 2),
    ('xz', (5, 10, 3), 2),
    ('yx', (10, 3), 2),
])
def test_projector_string(projection_str, in_coords_shape, out_coords_shape_end):
    coords = np.random.rand(*in_coords_shape)
    projector = Projector(projection_str)
    projected = projector.project(coords)
    assert projected.shape == (*in_coords_shape[:-1], out_coords_shape_end)

@pytest.mark.parametrize("invalid_proj", ["x", "y", "z", "ab", "12", "xya"])
def test_projector_invalid_string(invalid_proj):
    with pytest.raises(ValueError):
        Projector(invalid_proj)

def test_projector_numpy():
    coords = np.random.rand(10, 3)
    projection_matrix = np.random.rand(3, 2)
    projector = Projector(projection_matrix)
    projected = projector.project(coords)
    assert projected.shape == (10, 2)
    np.testing.assert_allclose(projected, coords @ projection_matrix)

def test_projector_invalid_type():
    with pytest.raises(ValueError):
        Projector(123)

# Tests for _normalize_trajectories
def test_normalize_trajectories_single(single_trajectory_3d_np):
    result = _normalize_trajectories(single_trajectory_3d_np)
    assert isinstance(result, list)
    assert len(result) == 1
    assert np.array_equal(result[0], single_trajectory_3d_np)

def test_normalize_trajectories_batch(batch_trajectory_3d_np):
    result = _normalize_trajectories(batch_trajectory_3d_np)
    assert isinstance(result, list)
    assert len(result) == batch_trajectory_3d_np.shape[0]
    for i in range(batch_trajectory_3d_np.shape[0]):
        assert np.array_equal(result[i], batch_trajectory_3d_np[i])

def test_normalize_trajectories_list(ragged_trajectories_np):
    result = _normalize_trajectories(ragged_trajectories_np)
    assert result == ragged_trajectories_np

def test_normalize_trajectories_ragged_object_array(ragged_trajectories_np):
    ragged_array = np.array(ragged_trajectories_np, dtype=object)
    result = _normalize_trajectories(ragged_array)
    assert isinstance(result, list)
    assert len(result) == len(ragged_trajectories_np)
    for i in range(len(ragged_trajectories_np)):
        assert np.array_equal(result[i], ragged_trajectories_np[i])

def test_normalize_trajectories_unsupported_type():
    with pytest.raises(TypeError):
        _normalize_trajectories(123)
        
def test_normalize_trajectories_unsupported_ndim():
    with pytest.raises(ValueError):
        _normalize_trajectories(np.random.rand(2, 2, 2, 2))
