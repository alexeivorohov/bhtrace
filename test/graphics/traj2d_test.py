import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import re

from bhtrace.graphics import plot2d
from bhtrace.scenarios.makers import make_schwarzschild
from bhtrace import Trajectory

OUTPUT_DIR = 'test/outputs' # <- this variable can control test data saving (None if no save)
BATCH_SIZE = 16
NSTEPS = 64

# convention: plots are tested in this class, not in Trajectory

def _get_filename(filename: str, request: pytest.FixtureRequest):
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

# --- Data fixtures ---
# add colors?

@pytest.fixture(scope="session")
def spiral_2d():
    t = np.linspace(0, 4 * np.pi, NSTEPS)
    x = t * np.cos(t)
    y = t * np.sin(t)
    return np.stack([x, y], axis=-1)


@pytest.fixture(scope="session")
def spiral_2d_batched():
    x0 = torch.linspace(16.0, 16.0, BATCH_SIZE).reshape(-1, 1)
    y0 = torch.linspace(0.0, 16.0, BATCH_SIZE).reshape(-1, 1)
    t = np.linspace(0, 4 * np.pi, NSTEPS).reshape(1, -1)
    x = x0 - t * np.cos(t)
    y = y0 - t * np.sin(t)
    return np.stack([x, y], axis=-1)


@pytest.fixture(scope="session")
def schwarzschild_2d() -> 'Trajectory':
    return make_schwarzschild('line')


# --- Call parameters ---

# populate by AI
basic_params = [
    {},
    {},
    {},
]

# populate by AI
colored_params = [
    {},
    {},
]

# --- Direct tests ---

@pytest.mark.parametrize('backend', ['mpl'])
@pytest.mark.parametrize('params', basic_params)
def test_spiral_2d(spiral_2d, backend, params, request):
    
    fig, ax = plot2d(spiral_2d, backend=backend, **params)

    if backend == 'mpl':
        fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
        plt.close(fig)


@pytest.mark.parametrize('backend', ['mpl'])
@pytest.mark.parametrize('params', basic_params)
def test_spiral_2d_batched(spiral_2d_batched, backend, params, request):

    fig, ax = plot2d(spiral_2d_batched, backend=backend, **params)

    if backend == 'mpl':
        fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
        plt.close(fig)

    
# --- Backend-specific tests ---

# TBW

# --- Trajectory tests - can be now moved to Trajectory class? ---

@pytest.mark.parametrize('backend', ['mpl'])
@pytest.mark.parametrize('params', basic_params)
def test_schwarzschild_2d(schwarzschild_2d: Trajectory, backend, params, request):

    info_text = f'backend: {backend}, params: {params}'
    fig, ax = schwarzschild_2d.plot2d(backend=backend, info_text=info_text, **params,)

    if backend == 'mpl':
        fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
        plt.close(fig)

@pytest.mark.parametrize('backend', ['mpl'])
@pytest.mark.parametrize('params', colored_params)
def test_schwarzschild_2d_colored(schwarzschild_2d: Trajectory, backend, params, request):
    info_text = f'backend: {backend}, params: {params}'
    fig, ax = schwarzschild_2d.plot2d(backend=backend, info_text=info_text, **params,)

    if backend == 'mpl':
        fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
        plt.close(fig)

# --- Old code ---

# # Tests for _plot_traj_2d
# def test_plot_traj_2d_single(single_trajectory_3d_np, request):
#     fig, ax = plt.subplots()
#     fig, ax = traj2d._plot_traj_2d(single_trajectory_3d_np, ax=ax, fig=fig, color='blue', label='test')
#     assert len(ax.lines) == 1
#     fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
#     plt.close(fig)

# def test_plot_traj_2d_batch(batch_trajectory_3d_np, request):
#     fig, ax = plt.subplots()
#     fig, ax = traj2d._plot_traj_2d(batch_trajectory_3d_np, ax=ax, fig=fig, color='green')
#     assert len(ax.lines) == 5 # 5 trajectories in the batch
#     fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
#     plt.close(fig)

# def test_plot_traj_2d_with_q(batch_trajectory_3d_np, request):
#     q = [np.random.rand(t.shape[0]) for t in batch_trajectory_3d_np]
#     fig, ax = plt.subplots()
#     _, ax = traj2d._plot_traj_2d(batch_trajectory_3d_np, q=q, cmap='viridis', label='test', ax=ax, fig=fig)
#     assert len(ax.collections) == 5
#     assert isinstance(ax.collections[0], LineCollection)
#     fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
#     plt.close(fig)

# # Tests for plot
# @pytest.mark.parametrize("color, label, kwargs", [
#     ("red", "Trajectory", {}),
#     (None, None, {"linewidth": 2}),
# ])
# def test_plot_simple(single_trajectory_3d_np, color, label, kwargs, request):
#     fig, ax = traj2d.plot(single_trajectory_3d_np, color=color, label=label, **kwargs)
#     assert isinstance(fig, plt.Figure)
#     assert isinstance(ax, plt.Axes)
#     assert len(ax.lines) == 1
#     if label:
#         assert ax.get_legend() is not None
#     if color:
#         assert ax.lines[0].get_color() == color
#     if "linewidth" in kwargs:
#         assert ax.lines[0].get_linewidth() == kwargs["linewidth"]
#     fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
#     plt.close(fig)

# def test_plot_with_text(single_trajectory_3d_np, request):
#     fig, ax = traj2d.plot2d(single_trajectory_3d_np, info_text="Info")
#     assert "Info" in [t.get_text() for t in fig.texts]
#     fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
#     plt.close(fig)

# def test_plot_with_q_and_colormap(single_trajectory_3d_np, request):
#     q = np.linspace(0, 1, 100)
#     fig, ax = traj2d.plot2d(single_trajectory_3d_np, q=q, q_label="Time", cmap="magma")
#     assert len(ax.collections) == 1
#     fig.savefig(f"{OUTPUT_DIR}/{_get_filename(__file__, request)}")
#     plt.close(fig)


# if __name__ == '__main__':
#     t = single_trajectory_3d_np()
#     plt.plot(t[..., 0], t[..., 1])
#     plt.show()
#     print(t.shape)
#     t = batch_trajectory_3d_np()
#     plt.plot(t[..., 0], t[..., 1])
#     plt.show()
#     print(t.shape)
#     t = ragged_trajectories_np()
#     for _t in t:
#         print(_t.shape)
#         plt.plot(_t[..., 0], _t[..., 1])
#         plt.show()
