import unittest
import torch
import os
import pickle
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bhtrace.tracing.ptracer import PTracer
from bhtrace.geometry import Photon, MinkowskiCart

class TestPTracer(unittest.TestCase):

    def setUp(self):
        self.spacetime = MinkowskiCart() # Metric is diag(-1, 1, 1, 1)
        self.particle = Photon(self.spacetime)
        self.tracer = PTracer(ode_method='RK4')
        # Initial position: t=0, x=0, y=1, z=0
        self.X0 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        # Initial covariant momentum: p_t=1, p_x=-1, p_y=0, p_z=0
        self.P0 = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
        self.T = 1.0
        self.nsteps = 100 # Use more steps for better accuracy

    def test_forward_and_conservation(self):
        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
        
        # 1. Check output shapes
        self.assertEqual(X.shape, (self.nsteps + 1, self.X0.shape[0], 4))
        self.assertEqual(P.shape, (self.nsteps + 1, self.P0.shape[0], 4))

        # 2. Check momentum conservation in flat space
        initial_momentum = self.P0[0]
        final_momentum = P[-1, 0]
        self.assertTrue(torch.allclose(final_momentum, initial_momentum, atol=1e-5),
                        f"Momentum not conserved! Initial: {initial_momentum}, Final: {final_momentum}")

        # 3. Check Hamiltonian conservation (mass-shell constraint for photons H=0)
        # The `evaluation` method calculates g^uv p_u p_v, which should be 0
        # We need to manually set particle and spc for evaluation method
        self.tracer.particle = self.particle
        self.tracer.spc = self.spacetime
        hamiltonian_values = self.tracer.evaluation(None, X, P)
        self.assertTrue(torch.all(torch.abs(hamiltonian_values) < 1e-5),
                        f"Hamiltonian not conserved! Values: {hamiltonian_values}")
        
    def test_plot_trajectory(self):
        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)

        # Extract data for plotting (for the first particle in the batch)
        t = X[:, 0, 0].cpu().numpy()
        x = X[:, 0, 1].cpu().numpy()
        y = X[:, 0, 2].cpu().numpy()
        z = X[:, 0, 3].cpu().numpy()

        pt = P[:, 0, 0].cpu().numpy()
        px = P[:, 0, 1].cpu().numpy()
        py = P[:, 0, 2].cpu().numpy()
        pz = P[:, 0, 3].cpu().numpy()

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle('PTracer in Minkowski Spacetime')

        # 3D Trajectory Plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(x, y, z)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Trajectory (x, y, z)')
        ax1.scatter(self.X0[0,1], self.X0[0,2], self.X0[0,3], c='g', marker='o', label='Start')
        ax1.scatter(x[-1], y[-1], z[-1], c='r', marker='x', label='End')
        ax1.legend()

        # Phase Space Plot
        ax2 = fig.add_subplot(122)
        ax2.plot(t, pt, label='p_t')
        ax2.plot(t, px, label='p_x')
        ax2.plot(t, py, label='p_y')
        ax2.plot(t, pz, label='p_z')
        ax2.set_xlabel('Coordinate Time (t)')
        ax2.set_ylabel('Momentum Components')
        ax2.set_title('Phase Space Evolution')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(os.path.dirname(__file__), 'ptracer_minkowski_test.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"\nPlot saved to {save_path}")

    def test_term(self):
        # To test __term__, we need to manually set the particle and spc attributes
        # that are normally set inside the .forward() call.
        self.tracer.particle = self.particle
        self.tracer.spc = self.spacetime
        # Test if the __term__ function returns derivatives with the correct shape
        dX, dP = self.tracer.__term__(0.0, self.X0, self.P0)
        self.assertEqual(dX.shape, self.X0.shape)
        self.assertEqual(dP.shape, self.P0.shape)

    def test_save_and_load(self):
        # This test combines save and load logic and ensures file cleanup
        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
        directory = 'test_dir_ptracer'
        if os.path.exists(directory):
            shutil.rmtree(directory) # Clean up before test
        os.makedirs(directory, exist_ok=True)
        filename = 'test_ptracer.pkl'
        comment = 'test comment'

        full_path = self.tracer.save(filename, directory=directory, comment=comment)
        
        self.assertTrue(os.path.exists(full_path))

        with open(full_path, 'rb') as f:
            result = pickle.load(f)
        
        self.assertIn('X', result)
        self.assertIn('P', result)
        self.assertIn('comment', result)
        self.assertEqual(result['comment'], comment)
        self.assertTrue(torch.allclose(result['X'], X))

        # Cleanup
        shutil.rmtree(directory)

if __name__ == '__main__':
    unittest.main()
