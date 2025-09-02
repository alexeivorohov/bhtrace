import unittest
import torch
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bhtrace.tracing import PTracer
from bhtrace.geometry import Photon, SphericallySymmetric, KerrSchild

# Utility function to convert spherical coordinates to cartesian for plotting
def sph_to_cart(coords):
    """Converts a batch of spherical coordinates (t, r, theta, phi) to Cartesian (x, y, z)."""
    r = coords[..., 1]
    theta = coords[..., 2]
    phi = coords[..., 3]
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()

class TestSchwarzschildTracer(unittest.TestCase):

    def setUp(self):
        # Schwarzschild spacetimsude with M=1 (so event horizon is at r=2.0)
        self.spacetime = SphericallySymmetric() 
        self.particle = Photon(self.spacetime)
        self.tracer = PTracer(ode_method='RK4')
        
        # Initial conditions for an unstable circular photon orbit at r=3M
        r0 = 3.0
        # p_phi / p_t = r / sqrt(r-2) -> let p_t = -1, then p_phi = 3 / sqrt(1) = 3
        self.X0 = torch.tensor([[0.0, r0, torch.pi / 2.0, 0.0]]) # (t, r, theta, phi)
        self.P0 = torch.tensor([[-1.0, 0.0, 0.0, 3.0]]) # (p_t, p_r, p_theta, p_phi)
        
        self.T = 50.0 # Time for a few orbits
        self.nsteps = 100
        self.plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

    def test_conservation(self):
        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
        
        # Check Hamiltonian conservation (mass-shell constraint H=0 for photons)
        hamiltonian_values = self.tracer.evaluation(None, X, P)
        self.assertTrue(torch.all(torch.abs(hamiltonian_values) < 1e-4),
                        f"Hamiltonian not conserved! Values: {hamiltonian_values}")

    def test_plot_photon_sphere_orbit(self):

        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
        
        x, y, z = sph_to_cart(X[:, 0, :])
        r = X[:, 0, 1].cpu().numpy()
        t = X[:, 0, 0].cpu().numpy()

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle('PTracer in Schwarzschild Spacetime (Photon Sphere)')

        # 3D Trajectory Plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x, y, z, label='Photon Orbit')
        # Plot the event horizon
        u, v = torch.meshgrid(torch.linspace(0, 2 * torch.pi, 20), torch.linspace(0, torch.pi, 20))
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

        # Radius vs Time Plot
        ax2 = fig.add_subplot(122)
        ax2.plot(t, r)
        ax2.axhline(y=3.0, color='r', linestyle='--', label='r=3M')
        ax2.set_xlabel('Coordinate Time (t)')
        ax2.set_ylabel('Radius (r)')
        ax2.set_title('Orbital Radius')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.plot_dir, 'schwarzschild_photon_sphere.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"\nPlot saved to {save_path}")

class TestKerrTracer(unittest.TestCase):

    def setUp(self):
        # Kerr spacetime with a=0.9M
        self.spacetime = KerrSchild(a=0.9)
        self.particle = Photon(self.spacetime)
        self.tracer = PTracer(ode_method='RK4')

        # Initial conditions for a photon in the equatorial plane
        self.X0 = torch.tensor([[0.0, 10.0, 0.1, 0.0]]) # Start far away, slightly off-axis
        # Fire towards the black hole
        p_t = 1.0
        p_x = -1.0
        p_y = 0.05
        p_z = 0.0
        # Normalize for null geodesic in nearly flat space
        p_t = (p_x**2 + p_y**2 + p_z**2)**0.5
        self.P0 = torch.tensor([[p_t, p_x, p_y, p_z]])
        
        self.T = 30.0
        self.nsteps = 2000
        self.plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

    def test_conservation(self):
        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
        
        # Check Hamiltonian conservation (mass-shell constraint H=0 for photons)
        hamiltonian_values = self.tracer.evaluation(None, X, P)
        self.assertTrue(torch.all(torch.abs(hamiltonian_values) < 1e-4),
                        f"Hamiltonian not conserved! Values: {hamiltonian_values}")

    def test_plot_kerr_orbit(self):
        
        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
        
        x = X[:, 0, 1].cpu().numpy()
        y = X[:, 0, 2].cpu().numpy()
        z = X[:, 0, 3].cpu().numpy()
        t = X[:, 0, 0].cpu().numpy()

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle('PTracer in Kerr Spacetime (a=0.9)')

        # 2D Trajectory Plot (x-y plane)
        ax1 = fig.add_subplot(121)
        ax1.plot(x, y, label='Photon Path')
        ax1.scatter([0], [0], c='k', marker='o', label='Singularity')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Trajectory in Equatorial Plane')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal', adjustable='box')

        # Z vs Time Plot
        ax2 = fig.add_subplot(122)
        ax2.plot(t, z)
        ax2.set_xlabel('Coordinate Time (t)')
        ax2.set_ylabel('Z coordinate')
        ax2.set_title('Deviation from Equatorial Plane')
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.plot_dir, 'kerr_orbit_test.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"\nPlot saved to {save_path}")

if __name__ == '__main__':
    unittest.main()
