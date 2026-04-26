import pytest
import torch
import os
import pickle
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bhtrace.tracing.ptracer import PTracer
from bhtrace.geometry import Photon
from bhtrace.geometry.spacetime import MinkowskiCart, SphericallySymmetric
from bhtrace.utils.transform import sph2cart

def mpl_plot_schwarzschild(X, P, save_path=None):
    '''
    Plots the trajectory of a particle in Schwarzschild spacetime.
    '''
    x, y, z = sph2cart(X[:, 0, 1], X[:, 0, 2], X[:, 0, 3])
    r = X[:, 0, 1].cpu().numpy()
    t = X[:, 0, 0].cpu().numpy()

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('PTracer in Schwarzschild Spacetime (Photon Sphere)')

    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x, y, z, label='Photon Orbit')
    # Plot the event horizon
    u, v = torch.meshgrid(torch.linspace(0, 2 * torch.pi, 20), torch.linspace(0, torch.pi, 20), indexing='xy')
    horizon_x = 2.0 * torch.cos(u) * torch.sin(v)
    horizon_y = 2.0 * torch.sin(u) * torch.sin(v)
    horizon_z = 2.0 * torch.cos(v)
    ax1.plot_surface(horizon_x, horizon_y, horizon_z, color='k', alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.view_init(elev=30., azim=45)
    ax1.set_box_aspect([1,1,1])

    # 2. Radius vs Time Plot
    ax2 = fig.add_subplot(122)
    ax2.plot(t, r)
    ax2.axhline(y=3.0, color='r', linestyle='--', label='r=3M')
    ax2.set_xlabel('Coordinate Time (t)')
    ax2.set_ylabel('Radius (r)')
    ax2.set_title('Orbital Radius')
    ax2.legend()
    ax2.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()

    return fig

@pytest.fixture
def minkowski_setup():
    spacetime = MinkowskiCart()
    particle = Photon(spacetime)
    # Initial position: t=0, x=0, y=1, z=0
    X0 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    # Initial covariant momentum: p_t=1, p_x=-1, p_y=0, p_z=0
    P0 = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
    T = 1.0
    nsteps = 100 # Use more steps for better accuracy
    return {
        "spacetime": spacetime,
        "particle": particle,
        "X0": X0,
        "P0": P0,
        "T": T,
        "nsteps": nsteps,
    }

@pytest.fixture
def schwarzschild_setup():
    spacetime = SphericallySymmetric()
    particle = Photon(spacetime)
    # Initial conditions for a circular orbit at r=3M
    X0 = torch.tensor([[0.0, 3.0, torch.pi/2, 0.0]]) # t, r, theta, phi
    P0 = torch.tensor([[1.0, 0.0, 0.0, 3.0 * torch.sqrt(torch.tensor(3.0))]]) # p_t, p_r, p_theta, p_phi
    T = 200.0
    nsteps = 2000
    return {
        "spacetime": spacetime,
        "particle": particle,
        "X0": X0,
        "P0": P0,
        "T": T,
        "nsteps": nsteps,
    }

@pytest.mark.parametrize("ode_method, atol", [("RK4", 1e-5), ("Leapfrog", 1e-4), ("Euler", 1e-2)])
def test_forward_and_conservation(minkowski_setup, ode_method, atol):
    particle = minkowski_setup["particle"]
    X0 = minkowski_setup["X0"]
    P0 = minkowski_setup["P0"]
    T = minkowski_setup["T"]
    nsteps = minkowski_setup["nsteps"]

    tracer = PTracer(ode_method=ode_method)
    traj = tracer.forward(particle, X0, P0, T, nsteps)
    X, P = traj.X, traj.P
    
    assert X.shape == (nsteps + 1, X0.shape[0], 4)
    assert P.shape == (nsteps + 1, P0.shape[0], 4)

    initial_momentum = P0[0]
    final_momentum = P[-1, 0]
    assert torch.allclose(final_momentum, initial_momentum, atol=atol), f"Momentum not conserved with {ode_method}!"

    hamiltonian_values = particle.hmlt(X, P)
    assert torch.all(torch.abs(hamiltonian_values) < atol), f"Hamiltonian not conserved with {ode_method}!"
    
def test_term(minkowski_setup):
    particle = minkowski_setup["particle"]
    tracer = PTracer()
    tracer.particle = particle
    tracer.spc = minkowski_setup["spacetime"]
    dX, dP = tracer.__term__(0.0, minkowski_setup["X0"], minkowski_setup["P0"])
    assert dX.shape == minkowski_setup["X0"].shape
    assert dP.shape == minkowski_setup["P0"].shape

# def test_save_and_load(minkowski_setup):
#     particle = minkowski_setup["particle"]
#     X0 = minkowski_setup["X0"]
#     P0 = minkowski_setup["P0"]
#     T = minkowski_setup["T"]
#     nsteps = minkowski_setup["nsteps"]

#     tracer = PTracer()
#     traj = tracer.forward(particle, X0, P0, T, nsteps)

#     directory = 'test_dir_ptracer'
#     if os.path.exists(directory):
#         shutil.rmtree(directory)
#     os.makedirs(directory, exist_ok=True)
#     filename = 'test_ptracer.pkl'
#     comment = 'test comment'

#     full_path = traj.save(filename, directory=directory, comment=comment)
    
#     assert os.path.exists(full_path)

#     with open(full_path, 'rb') as f:
#         result = pickle.load(f)
    
#     assert 'X' in result
#     assert 'P' in result
#     assert 'comment' in result
#     assert result['comment'] == comment
#     assert torch.allclose(result['X'], traj.X)

#     shutil.rmtree(directory)

def test_photon_sphere_orbit(schwarzschild_setup):
    particle = schwarzschild_setup["particle"]
    X0 = schwarzschild_setup["X0"]
    P0 = schwarzschild_setup["P0"]
    T = schwarzschild_setup["T"]
    nsteps = schwarzschild_setup["nsteps"]
    
    tracer = PTracer(ode_method='RK4')
    traj = tracer.forward(particle, X0, P0, T, nsteps)
    X, P = traj.X, traj.P

    final_radius = X[-1, 0, 1]
    assert final_radius.item() == pytest.approx(3.0, abs=1e-2)

    save_path = os.path.join(os.path.dirname(__file__), 'ptracer_schwarzschild_test.png')
    mpl_plot_schwarzschild(X, P, save_path=save_path)
