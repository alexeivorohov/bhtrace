import unittest
import torch
import pickle

import sys
import os
root_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.tracing.ptracer import PTracer
from bhtrace.geometry import Photon, MinkowskiCart

class TestPTracer(unittest.TestCase):

    def setUp(self):

        self.spacetime = MinkowskiCart()
        self.particle = Photon(self.spacetime)
        self.tracer = PTracer()
        self.X0 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        self.P0 = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
        self.T = 1.0
        self.nsteps = 10


    def test_forward(self):

        X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
        self.assertEqual(X.shape, (self.nsteps, self.X0.shape[0], 4))
        self.assertEqual(P.shape, (self.nsteps, self.X0.shape[0], 4))

        # Check that the particle follows a straight line
        initial_position = self.X0[0]
        initial_momentum = self.P0[0]
        final_position = X[-1, 0]
        final_momentum = P[-1, 0]

        # Calculate expected final position based on straight-line motion
        expected_final_position = initial_position + self.T * initial_momentum

        self.assertTrue(torch.allclose(final_position, expected_final_position, atol=1e-5))
        self.assertTrue(torch.allclose(final_momentum, initial_momentum, atol=1e-5))


    # def test_evnt(self):

    #     XP = torch.cat((self.X0[0], self.P0[0]))
    #     t = 0.0
    #     result = self.tracer.evnt(t, XP)
    #     self.assertIsInstance(result, torch.Tensor)


    # def test_reg(self):

    #     XP = torch.cat((self.X0[0], self.P0[0]))
    #     t = 0.0
    #     result = self.tracer.reg(t, XP)
    #     self.assertEqual(result.shape, XP.shape)


    # def test_term(self):

    #     XP = torch.cat((self.X0[0], self.P0[0]))
    #     t = 0.0
    #     result = self.tracer.__term__(t, XP)
    #     self.assertEqual(result.shape, XP.shape)


    # def test_save(self):

    #     X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
    #     filename = 'test_ptracer.pkl'
    #     self.tracer.save(filename)
    #     self.assertTrue(os.path.exists(filename))
    #     with open(filename, 'rb') as f:
    #         result = pickle.load(f)
    #     self.assertIn('X', result)
    #     self.assertIn('P', result)
    #     os.remove(filename)


    # def test_save_with_directory(self):

    #     X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
    #     directory = 'test_dir'
    #     os.makedirs(directory, exist_ok=True)
    #     filename = 'test_ptracer.pkl'
    #     full_path = self.tracer.save(filename, directory)
    #     self.assertTrue(os.path.exists(full_path))
    #     with open(full_path, 'rb') as f:
    #         result = pickle.load(f)
    #     self.assertIn('X', result)
    #     self.assertIn('P', result)
    #     os.remove(full_path)
    #     os.rmdir(directory)


    # def test_save_with_comment(self):

    #     X, P = self.tracer.forward(self.particle, self.X0, self.P0, self.T, self.nsteps)
    #     filename = 'test_ptracer.pkl'
    #     comment = 'Test comment'
    #     self.tracer.save(filename, comment=comment)
    #     self.assertTrue(os.path.exists(filename))
    #     with open(filename, 'rb') as f:
    #         result = pickle.load(f)
    #     self.assertIn('comment', result)
    #     self.assertEqual(result['comment'], comment)
    #     os.remove(filename)


if __name__ == '__main__':
    unittest.main()