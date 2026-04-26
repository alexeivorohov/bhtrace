import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import re

import bhtrace.graphics.traj2d as traj2d
from bhtrace.graphics.traj2d import Projector, _normalize_trajectories

OUTPUT_DIR = 'test/outputs'

def _get_filename(request):
    """Generate a sanitized filename from the pytest request object."""
    module_name = os.path.basename(request.fspath).replace('_test.py', '')
    
    test_case_name = request.node.name
    test_case_name = re.sub(r'\[', '_', test_case_name)
    test_case_name = re.sub(r'\]', '', test_case_name)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', test_case_name)
    
    return f"{module_name}_{sanitized_name}.png"

@pytest.fixture(scope="session", autouse=True)
def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixtures for test data
@pytest.fixture
def single_trajectory_3d_np():
    """Returns a single continuous 3D trajectory (a spiral) as numpy array."""
    t = np.linspace(0, 4 * np.pi, 100)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    return np.stack([x, y, z], axis=1)

@pytest.fixture
def single_trajectory_3d_torch():
    """Returns a single continuous 3D trajectory (a spiral) as torch tensor."""
    t = torch.linspace(0, 4 * np.pi, 100)
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    z = t
    return torch.stack([x, y, z], dim=1)

@pytest.fixture
def batch_trajectory_3d_np():
    """Returns a batch of continuous 3D trajectories as numpy array."""
    t = np.linspace(0, 4 * np.pi, 100)
    trajectories = []
    for i in range(5):
        # Varying spirals
        x = t * np.cos(t + i * np.pi / 4)
        y = t * np.sin(t + i * np.pi / 4)
        z = t + i
        trajectories.append(np.stack([x, y, z], axis=1))
    return np.stack(trajectories)
    
@pytest.fixture
def batch_trajectory_3d_torch():
    """Returns a batch of continuous 3D trajectories as torch tensor."""
    t = torch.linspace(0, 4 * np.pi, 100)
    trajectories = []
    for i in range(5):
        x = t * torch.cos(t + i * np.pi / 4)
        y = t * torch.sin(t + i * np.pi / 4)
        z = t + i
        trajectories.append(torch.stack([x, y, z], dim=1))
    return torch.stack(trajectories)

@pytest.fixture
def multiple_trajectories_torch():
    """Returns a list of continuous trajectories as torch tensors."""
    trajectories = []
    for i in range(3):
        t = torch.linspace(0, np.random.uniform(2, 6) * np.pi, 100)
        x = (t + i * 2) * torch.cos(t)
        y = (t + i * 2) * torch.sin(t)
        z = t
        trajectories.append(torch.stack([x, y, z], dim=1))
    return trajectories

@pytest.fixture
def ragged_trajectories_np():
    """Returns a list of continuous trajectories of different lengths as numpy arrays."""
    trajectories = []
    for i in range(3):
        t = np.linspace(0, np.random.uniform(2, 6) * np.pi, 50 + i*20)
        x = (t + i * 2) * np.cos(t)
        y = (t + i * 2) * np.sin(t)
        z = t
        trajectories.append(np.stack([x, y, z], axis=1))
    return trajectories

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

# Tests for _plot_traj_2d
def test_plot_traj_2d_single(single_trajectory_3d_np, request):
    fig, ax = plt.subplots()
    fig, ax = traj2d._plot_traj_2d(single_trajectory_3d_np, ax=ax, fig=fig, color='blue', label='test')
    assert len(ax.lines) == 1
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_traj_2d_batch(batch_trajectory_3d_np, request):
    fig, ax = plt.subplots()
    fig, ax = traj2d._plot_traj_2d(batch_trajectory_3d_np, ax=ax, fig=fig, color='green')
    assert len(ax.lines) == 5 # 5 trajectories in the batch
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_traj_2d_with_q(batch_trajectory_3d_np, request):
    q = [np.random.rand(t.shape[0]) for t in batch_trajectory_3d_np]
    fig, ax = plt.subplots()
    _, ax = traj2d._plot_traj_2d(batch_trajectory_3d_np, q=q, cmap='viridis', label='test', ax=ax, fig=fig)
    assert len(ax.collections) == 5
    assert isinstance(ax.collections[0], LineCollection)
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

# Tests for plot
@pytest.mark.parametrize("color, label, kwargs", [
    ("red", "Trajectory", {}),
    (None, None, {"linewidth": 2}),
])
def test_plot_simple(single_trajectory_3d_np, color, label, kwargs, request):
    fig, ax = traj2d.plot(single_trajectory_3d_np, color=color, label=label, **kwargs)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 1
    if label:
        assert ax.get_legend() is not None
    if color:
        assert ax.lines[0].get_color() == color
    if "linewidth" in kwargs:
        assert ax.lines[0].get_linewidth() == kwargs["linewidth"]
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_with_text(single_trajectory_3d_np, request):
    fig, ax = traj2d.plot(single_trajectory_3d_np, info_text="Info")
    assert "Info" in [t.get_text() for t in fig.texts]
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_with_q_and_colormap(single_trajectory_3d_np, request):
    q = np.linspace(0, 1, 100)
    fig, ax = traj2d.plot(single_trajectory_3d_np, q=q, q_label="Time", cmap="magma")
    assert len(ax.collections) == 1
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

# Tests for plot_multiple
def test_plot_multiple(multiple_trajectories_torch, request):
    labels = [f"T{i}" for i in range(len(multiple_trajectories_torch))]
    colors = ['r', 'g', 'b']
    fig, ax = traj2d.plot_multiple(multiple_trajectories_torch, labels=labels, colors=colors, legend=True)
    assert len(ax.lines) == len(multiple_trajectories_torch)
    assert ax.get_legend() is not None
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_multiple_with_q(multiple_trajectories_torch, request):
    q_data = [torch.rand(t.shape[0]) for t in multiple_trajectories_torch]
    
    # Test with a single colormap name
    fig, ax = traj2d.plot_multiple(multiple_trajectories_torch, q=q_data, q_label="Q Value", colors='viridis')
    assert len(ax.collections) == len(multiple_trajectories_torch)
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}-single_cmap.png")
    plt.close(fig)

    # Test with a list of colormap names
    colors = ['viridis', 'plasma', 'inferno']
    fig, ax = traj2d.plot_multiple(multiple_trajectories_torch, q=q_data, colors=colors, legend=True)
    assert len(ax.collections) == len(multiple_trajectories_torch)
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}-multi_cmap.png")
    plt.close(fig)

def test_plot_multiple_with_horizon(multiple_trajectories_torch, request):
    fig, ax = traj2d.plot_multiple(multiple_trajectories_torch, horizon=2.0)
    assert len(ax.patches) == 1
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)


# Tests for plot_mosaic
@pytest.mark.skip(reason="WIP")
def test_plot_mosaic(multiple_trajectories_torch, request):
    labels = ['A', 'B', 'C']
    fig, axs = traj2d.plot_mosaic(multiple_trajectories_torch, labels=labels, color='blue')
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(axs, dict)
    assert set(axs.keys()) == set(labels)
    for label in labels:
        ax = axs[label]
        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) > 0
        assert ax.lines[0].get_color() == 'blue'
        assert ax.get_title() == label
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

@pytest.mark.skip(reason="WIP")
def test_plot_mosaic_no_labels(multiple_trajectories_torch, request):
    fig, axs = traj2d.plot_mosaic(multiple_trajectories_torch)
    expected_labels = [str(i) for i in range(len(multiple_trajectories_torch))]
    assert set(axs.keys()) == set(expected_labels)
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

@pytest.mark.skip(reason="WIP")
def test_plot_mosaic_customized(multiple_trajectories_torch, request):
    labels = ['xy', 'yz', 'xz']
    projections = ['xy', 'yz', 'xz']
    q_values = [torch.linspace(0, 1, 100) for _ in labels]
    q_labels = ["Red Q", "Green Q", "Blue Q"]
    cmaps = ['Reds', 'Greens', 'Blues']

    fig, axs = traj2d.plot_mosaic(
        multiple_trajectories_torch,
        labels=labels,
        projection_list=projections,
        q_list=q_values,
        q_label_list=q_labels,
        color_list=cmaps
    )

    assert set(axs.keys()) == set(labels)
    
    for i, label in enumerate(labels):
      ax = axs[label]
      assert len(ax.collections) == 1
      assert ax.get_xlabel() == 'x'
      assert ax.get_ylabel() == 'y'
    
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)
