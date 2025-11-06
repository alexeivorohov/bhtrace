import unittest
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.quiver import Quiver
from bhtrace.graphics.plot3d import Plot3D

class TestPlot3D(unittest.TestCase):

    def setUp(self):
        plt.close('all')

    def test_fig_handler_no_args(self):
        fig, ax = Plot3D.fig_handler()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, 'get_zlim'))

    def test_fig_handler_with_fig(self):
        fig = plt.figure()
        fig_new, ax = Plot3D.fig_handler(fig=fig)
        self.assertIs(fig, fig_new)
        self.assertIsNotNone(ax)
        self.assertIs(fig, ax.get_figure())

    def test_fig_handler_with_ax(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig_new, ax_new = Plot3D.fig_handler(ax=ax)
        self.assertIs(fig, fig_new)
        self.assertIs(ax, ax_new)

    def test_dir_handler(self):
        fig, ax = Plot3D.fig_handler()
        Plot3D.dir_handler(ax=ax, elev=30, azim=60)
        self.assertEqual(ax.elev, 30)
        self.assertEqual(ax.azim, 60)

    def test_cmap_handler(self):
        values = torch.linspace(0, 1, 10)
        colors, cmap, norm = Plot3D.cmap_handler(values)
        self.assertIsNotNone(colors)
        self.assertIsNotNone(cmap)
        self.assertIsNotNone(norm)
        self.assertEqual(colors.shape[0], values.shape[0])
        self.assertEqual(colors.shape[1], 4) # RGBA

    def test_point_cloud_with_values(self):
        points = torch.rand(10, 3)
        values = torch.rand(10)
        fig, ax = Plot3D.point_cloud(points, values=values)
        self.assertEqual(len(ax.collections), 1)
        scatter = ax.collections[0]
        self.assertIsNotNone(scatter.get_cmap())
        self.assertTrue(scatter.get_array() is not None)

    def test_point_cloud_with_view(self):
        points = torch.rand(10, 3)
        fig, ax = Plot3D.point_cloud(points, elev=45, azim=45)
        self.assertEqual(ax.elev, 45)
        self.assertEqual(ax.azim, 45)

    def test_lines_with_values(self):
        points = torch.rand(5, 10, 3)
        values = torch.rand(5) # Per-line values
        fig, ax = Plot3D.lines(points, values=values)
        self.assertGreater(len(ax.lines), 0)
        # Check if lines have different colors
        colors = [line.get_color() for line in ax.lines]
        unique_colors = set(map(str, colors))
        self.assertGreater(len(unique_colors), 1)

    def test_lines_with_view(self):
        points = torch.rand(5, 10, 3)
        fig, ax = Plot3D.lines(points, elev=20, azim=70)
        self.assertEqual(ax.elev, 20)
        self.assertEqual(ax.azim, 70)

    def test_vector_field_with_values(self):
        points = torch.rand(10, 3)
        vectors = torch.rand(10, 3)
        values = torch.rand(10)
        try:
            fig, ax = Plot3D.vector_field(points, vectors, values=values)
            # If it runs without error, we consider it a pass for now.
        except Exception as e:
            self.fail(f"vector_field with values failed with {e}")

    def test_vector_field_with_view(self):
        points = torch.rand(10, 3)
        vectors = torch.rand(10, 3)
        fig, ax = Plot3D.vector_field(points, vectors, elev=10, azim=80)
        self.assertEqual(ax.elev, 10)
        self.assertEqual(ax.azim, 80)

if __name__ == '__main__':
    unittest.main()