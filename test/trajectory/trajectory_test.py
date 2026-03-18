import unittest
import torch
import os

from bhtrace import Trajectory
from bhtrace.tracing import MockTracer
from bhtrace.geometry import Photon, SphericallySymmetric

class TestTrajectory(unittest.TestCase):

    def setUp(self):
        self.spacetime = SphericallySymmetric()
        self.particle = Photon(self.spacetime)
        self.tracer = MockTracer(self.particle, self.spacetime)

        X0 = torch.tensor([[0.0, 3.0, torch.pi / 2.0, 0.0]])
        P0 = torch.tensor([[-1.0, 0.0, 0.0, 3.0]])

        self.trajectory = self.tracer.forward(self.particle, X0, P0, 10.0, 100)

        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)

    def test_save_load(self):
        filename = os.path.join(self.test_dir, 'test_trajectory.pth')
        self.trajectory.save(filename)
        self.assertTrue(os.path.exists(filename))

        loaded_trajectory = Trajectory.load(filename)
        self.assertIsInstance(loaded_trajectory, Trajectory)
        self.assertTrue(torch.equal(self.trajectory.X, loaded_trajectory.X))
        self.assertTrue(torch.equal(self.trajectory.P, loaded_trajectory.P))

        # ToDo: Add proper deserialization logic and then uncomment the following lines
        # self.assertEqual(self.trajectory.particle_state, loaded_trajectory.particle_state)
        # self.assertEqual(self.trajectory.spacetime_state, loaded_trajectory.spacetime_state)

    def test_to(self):
        if torch.cuda.is_available():
            self.trajectory.to('cuda')
            self.assertEqual(self.trajectory.X.device.type, 'cuda')
            self.assertEqual(self.trajectory.P.device.type, 'cuda')

if __name__ == '__main__':
    unittest.main()
