import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re

import bhtrace.graphics as bhg
from bhtrace.scenarios.makers import make_schwarzschild

OUTPUT_DIR = 'test/outputs'

def _get_filename(request):
    """Generate a sanitized filename from the pytest request object."""
    module_name = os.path.basename(request.fspath).replace('_test.py', '')
    
    # Sanitize node name by removing invalid characters
    test_case_name = request.node.name
    test_case_name = re.sub(r'\[', '_', test_case_name)
    test_case_name = re.sub(r'\]', '', test_case_name)
    sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', test_case_name)
    
    return f"{module_name}_{sanitized_name}.png"

@pytest.fixture(scope="session", autouse=True)
def create_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

@pytest.fixture
def single_trajectory():
    """Returns a single continuous 3D trajectory (a helix)."""
    t = torch.linspace(0, 8 * np.pi, 200)
    x = torch.cos(t)
    y = torch.sin(t)
    z = t / 4
    return torch.stack([x, y, z], dim=1)

@pytest.fixture
def multiple_trajectories():
    """Returns a list of three continuous trajectories."""
    trajectories = []
    for i in range(3):
        t = torch.linspace(0, 8 * np.pi, 200)
        x = (i + 1) * torch.cos(t)
        y = (i + 1) * torch.sin(t)
        z = t / 4 + i
        trajectories.append(torch.stack([x, y, z], dim=1))
    return trajectories

@pytest.fixture(scope="session")
def spiral_3d_batched():
    """Returns a batch of continuous 3D trajectories as numpy array."""
    t = np.linspace(0, 4 * np.pi, NSTEPS)
    trajectories = []
    for i in range(BATCH_SIZE):
        # Varying spirals
        x = t * np.cos(t + i * np.pi / 4)
        y = t * np.sin(t + i * np.pi / 4)
        z = t + i
        trajectories.append(np.stack([x, y, z], axis=1))
    return np.stack(trajectories)
    
@pytest.fixture(scope="session")
def spiral_3d_ragged():
    """Returns a list of continuous trajectories of different lengths as numpy arrays."""
    trajectories = []
    for i in range(3):
        t = np.linspace(0, np.random.uniform(2, 6) * np.pi, NSTEPS + i*8)
        x = (t + i * 2) * np.cos(t)
        y = (t + i * 2) * np.sin(t)
        z = t
        trajectories.append(np.stack([x, y, z], axis=1))
    return trajectories


@pytest.fixture(scope="session")
def spiral_3d():
    """Returns a single continuous 3D trajectory (a spiral) as numpy array."""
    
    t = np.linspace(0, 4 * np.pi, NSTEPS)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    return np.stack([x, y, z], axis=1)

@pytest.fixture(scope="session")
def schwarzschild_3d() -> 'Trajectory':
    return make_schwarzschild('square')

@pytest.fixture
def points_and_vectors():
    """Returns a set of points and vectors for vector field plots."""
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.4),
                          np.arange(-0.8, 1, 0.4),
                          np.arange(-0.8, 1, 0.8))
    points = torch.from_numpy(np.stack([x, y, z], axis=-1).astype(np.float32))
    
    u = -points[..., 1]
    v = points[..., 0]
    w = torch.zeros_like(u)
    vectors = torch.stack([u, v, w], dim=-1)
    
    return points, vectors

def test_plot_single_trajectory(single_trajectory, request):
    """Test plotting a single 3D trajectory."""
    fig, ax = bhg.traj3d.plot(single_trajectory, color='c', label='Helix')
    assert len(ax.lines) == 1
    assert ax.get_legend() is not None
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_with_q_values(single_trajectory, request):
    """Test plotting a single 3D trajectory with gradient color."""
    q = torch.linspace(0, 1, single_trajectory.shape[0])
    fig, ax = bhg.traj3d.plot(single_trajectory, q=q, q_label='Time', color='viridis')
    assert len(ax.collections) == 1
    # Check that a colorbar was created
    assert len(fig.axes) > 1
    assert fig.axes[-1].get_ylabel() == 'Time'
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_multiple(multiple_trajectories, request):
    """Test plotting multiple trajectories on the same axes."""
    labels = ['Traj 1', 'Traj 2', 'Traj 3']
    colors = ['r', 'g', 'b']
    fig, ax = bhg.traj3d.plot_multiple(
        multiple_trajectories,
        labels=labels,
        color_list=colors
    )
    assert len(ax.lines) == len(multiple_trajectories)
    assert ax.get_legend() is not None
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_plot_mosaic(multiple_trajectories, request):
    """Test plotting trajectories on a mosaic of subplots."""
    labels = ['A', 'B', 'C']
    q_values = [torch.linspace(0, 1, t.shape[0]) for t in multiple_trajectories]
    q_labels = ["Time A", "Time B", "Time C"]
    cmaps = ['Reds', 'Greens', 'Blues']

    fig, axs = bhg.traj3d.plot_mosaic(
        multiple_trajectories,
        labels=labels,
        q_list=q_values,
        q_label_list=q_labels,
        color_list=cmaps,
        elev=20,
        azim=30
    )
    assert len(axs) == len(multiple_trajectories)
    # Check for colorbars (each plot call adds one)
    assert len(fig.axes) == 2 * len(multiple_trajectories)
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_point_cloud(single_trajectory, request):
    """Test plotting a 3D point cloud."""
    q = torch.linspace(0, 1, single_trajectory.shape[0])
    fig, ax = bhg.traj3d.point_cloud(single_trajectory, values=q, cmap='plasma')
    # Scatter returns a Path3DCollection, not lines
    assert len(ax.collections) > 0
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)

def test_vector_field(points_and_vectors, request):
    """Test plotting a 3D vector field."""
    points, vectors = points_and_vectors
    values = torch.norm(vectors, dim=-1)
    fig, ax = bhg.traj3d.vector_field(points, vectors, values=values, cmap='coolwarm', length=0.1)
    # Quiver returns a Quiver object
    assert len(ax.collections) > 0
    fig.savefig(f"{OUTPUT_DIR}/{_get_filename(request)}")
    plt.close(fig)