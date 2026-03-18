import unittest
import torch
import numpy as np

import sys
import os
root_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.utils.routines import EulerRotation, points_generate, net, bisection, def_fspace, levi_civita_tensor, print_status_bar, rotate_points_cloud

class TestRoutines(unittest.TestCase):

# Depercated
    # def test_cart2sph(self):
    #     inX = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    #     inP = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    #     outX, outP = cart2sph(inX, inP)
    #     self.assertEqual(outX.shape, inX.shape)
    #     self.assertEqual(outP.shape, inP.shape)

# Depercated
    # def test_sph2cart(self):
    #     inX = torch.tensor([[0.0, 1.0, np.pi / 2, 0.0]])
    #     inP = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    #     outX, outP = sph2cart(inX, inP)
    #     self.assertEqual(outX.shape, inX.shape)
    #     self.assertEqual(outP.shape, inP.shape)

    def test_points_generate(self):
        ts = [0.0, 1.0]
        rs = [1.0, 2.0]
        ths = [0.0, np.pi / 2]
        phs = [0.0, np.pi]
        X = points_generate(ts, rs, ths, phs)
        self.assertEqual(X.shape, (len(ts) * len(rs) * len(ths) * len(phs), 4))

    def test_net(self):
        xx, yy, zz = net(shape='square', rng=(5, 5), YZ0=[0, 0], X0=20, YZsize=[8, 8])
        self.assertEqual(xx.shape, yy.shape)
        self.assertEqual(yy.shape, zz.shape)

    def test_bisection(self):
        func = lambda x, par: x**2 - par
        x_min = torch.tensor([-1.0])
        x_max = torch.tensor([2.0])
        par = torch.tensor([1.0])
        zero = bisection(func, x_min, x_max, par)
        self.assertTrue(torch.allclose(zero, torch.tensor([1.0]), atol=1e-4))

    # Should be fixed and tested with more complex functions
    def test_def_fspace(self):
        def func(x, par):
            return torch.where(torch.abs(x) < par, x, torch.tensor(float('nan')))

        par = torch.tensor([1.0])
        x_min = -2 * par
        x_max = 2 * par

        x_min, x_max = def_fspace(func, x_min, x_max, par)
        self.assertTrue(torch.allclose(x_max, par, atol=1e-4))
        self.assertTrue(torch.allclose(x_min, -par, atol=1e-4))

    def test_levi_civita_tensor(self):
        dim = 3
        tensor = levi_civita_tensor(dim)
        self.assertEqual(tensor.shape, (dim, dim, dim))

    def test_print_status_bar(self):
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        print_status_bar(5, 10, 1.0)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn('50.00%', output)

    def test_EulerRotation_1vec(self):
        
        # Define input vectors and angles
        X = torch.tensor([1.0, 0.0, 0.0])
        
        dphi = torch.tensor(0.5)  # 30 degrees in radians
        dth = torch.tensor(0.5)   # 30 degrees in radians

        # Expected output after rotation
        expected_output = torch.tensor([[0.8660, 0.4330, 0.5],
                                        [0.4330, 0.8660, 0.5],
                                        [0.0, 0.0, 1.0]])

        # Perform rotation
        output = EulerRotation(X, dphi, dth)

        # Check if the output is close to the expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_EulerRotation_with_different_angles(self):
        # Test with different angles
        X = torch.tensor([[1.0, 0.0, 0.0]])
        dphi = torch.tensor(1.0)  # 57.3 degrees in radians
        dth = torch.tensor(1.0)   # 57.3 degrees in radians

        # Expected output after rotation
        expected_output = torch.tensor([[0.5, 0.5, 0.8660]])

        # Perform rotation
        output = EulerRotation(X, dphi, dth)

        # Check if the output is close to the expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


    def test_rotate_point_cloud(self):
        # Case 1: Simple 90-degree rotation
        points1 = torch.tensor([[10., 5., 2.]])
        dir_a1 = torch.tensor([1., 0., 0.])
        dir_b1 = torch.tensor([0., 1., 0.])
        rotated1 = rotate_points_cloud(points1, dir_a1, dir_b1)
        expected1 = torch.tensor([[-5., 10., 2.]])
        self.assertTrue(torch.allclose(rotated1, expected1, atol=1e-4))

        # Case 2: Identity rotation (a == b)
        points2 = torch.tensor([[1., 2., 3.]])
        dir_a2 = torch.tensor([0., 1., 0.])
        rotated2 = rotate_points_cloud(points2, dir_a2, dir_a2)
        self.assertTrue(torch.allclose(rotated2, points2, atol=1e-4))

        # Case 3: 180-degree rotation (anti-parallel)
        points3 = torch.tensor([[10., 5., 2.]])
        dir_a3 = torch.tensor([1., 0., 0.])
        dir_b3 = torch.tensor([-1., 0., 0.])
        rotated3 = rotate_points_cloud(points3, dir_a3, dir_b3)
        expected3 = torch.tensor([[-10., -5., 2.]])
        self.assertTrue(torch.allclose(rotated3, expected3, atol=1e-4))

        # Case 4: Rotation with additional angle
        points4 = torch.tensor([[10., 0., 0.]])
        dir_a4 = torch.tensor([1., 0., 0.])
        dir_b4 = torch.tensor([0., 1., 0.])
        angle4 = np.pi / 2.0
        rotated4 = rotate_points_cloud(points4, dir_a4, dir_b4, angle=angle4)
        expected4 = torch.tensor([[0., 0., 10.]])
        self.assertTrue(torch.allclose(rotated4, expected4, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
