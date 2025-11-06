import torch
import unittest

from bhtrace.utils.net import Linear, Rectangle, Circle, Hex

class TestNetConstruction(unittest.TestCase):

    def test_Linear(self):
        n_points = 10
        net = Linear(torch.tensor([20.0, 0.0, 0.0]), n_points)
        self.assertEqual(net.N, n_points)
        self.assertEqual(net.pos3d.shape, (n_points, 3))
        self.assertEqual(len(net.cells), n_points - 1)

    def test_Rectangle(self):
        shape = (4, 5)
        n_points = shape[0] * shape[1]
        net = Rectangle(torch.tensor([2.0, 2.0]), shape)
        self.assertEqual(net.N, n_points)
        self.assertEqual(net.pos3d.shape, (n_points, 3))
        self.assertEqual(len(net.cells), (shape[0] - 1) * (shape[1] - 1))

    def test_Circle(self):
        shape = (3, 8)  # 3 rings, 8 wedges
        n_points = 1 + shape[0] * shape[1]
        net = Circle(torch.tensor([10.0]), shape)
        self.assertEqual(net.N, n_points)
        self.assertEqual(net.pos3d.shape, (n_points, 3))
        # Cells: 8 triangles from center + 2 rings of 8 quads = 8 + 16 = 24
        # The cell calculation in the test is simplified, it should be:
        # n_wedges (for inner triangles) + (n_rings - 1) * n_wedges (for outer quads)
        expected_cells = shape[1] + (shape[0] - 1) * shape[1]
        self.assertEqual(len(net.cells), expected_cells)

    def test_Hex(self):
        shape = (4, 5)
        n_points = shape[0] * shape[1]
        net = Hex(torch.tensor([10.0, 10.0]), shape)
        self.assertEqual(net.N, n_points)
        self.assertEqual(net.pos3d.shape, (n_points, 3))
        # Cells: 2 triangles for each rhombus in a (n_q-1)x(n_r-1) grid
        expected_cells = 2 * (shape[0] - 1) * (shape[1] - 1)
        self.assertEqual(len(net.cells), expected_cells)

class TestNetOperations(unittest.TestCase):

    def setUp(self):
        self.net = Rectangle(torch.tensor([2.0, 2.0]), (3, 3))

    def test_upsample(self):
        n_initial = self.net.N
        self.assertGreater(n_initial, 0)
        
        # Upsample once
        net2 = self.net.upsample()
        self.assertGreater(net2.N, n_initial)
        self.assertFalse(net2.uniform)
        self.assertEqual(net2.gen, 1)

        # Upsample again
        net3 = net2.upsample()
        self.assertGreater(net3.N, net2.N)
        self.assertEqual(net3.gen, 2)

    def test_to_method(self):
        # Test dtype casting
        net_f64 = self.net.to(dtype=torch.float64)
        self.assertEqual(net_f64.pos3d.dtype, torch.float64)
        self.assertEqual(net_f64.weights.dtype, torch.float64)
        self.assertEqual(net_f64.X0.dtype, torch.float64)
        # Boolean tensor should not change dtype
        self.assertEqual(net_f64.traced.dtype, torch.bool)

        # Test moving to CPU device (as a basic check)
        net_cpu = self.net.to(device='cpu')
        self.assertEqual(str(net_cpu.pos3d.device), 'cpu')

        # Test moving to CUDA device ()

if __name__ == '__main__':
    unittest.main()
