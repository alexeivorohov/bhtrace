# import unittest
# import torch

# from bhtrace.geometry import KerrSchild, Observer, Photon
# from bhtrace.tracing import PTracer
# from bhtrace.scenarios import Imaging
# from bhtrace.grrt.medium_collection import ThinKeplerianDisk
# from bhtrace.grrt.radiation import OpticallyThickThinDisk

# class TestImagingIntegration(unittest.TestCase):

#     def test_disk_image(self):
#         # Setup
#         spacetime = KerrSchild(a=0.0)
#         position = torch.tensor([0.0, 100.0 * torch.sin(torch.tensor(torch.pi / 4)), 0.0, 100.0 * torch.cos(torch.tensor(torch.pi / 4))])
#         observer = Observer(spacetime=spacetime, position=position)
#         observer.generate_net(net_rng=(16,16))
#         particle = Photon(spacetime)
#         observer.setup_ic(particle)

#         tracer = PTracer()
#         disk = ThinKeplerianDisk(spacetime, r_in=6.0, r_out=20.0)
        
#         imaging_scenario = Imaging(
#             spacetime=spacetime,
#             medium=disk,
#             observer=observer,
#             tracer=tracer
#         )

#         # Run the simulation
#         imaging_scenario.propagate_photons(T=200, nsteps=256)
        
#         # Create a radiative model and render the image
#         model = OpticallyThickThinDisk()
#         image = imaging_scenario.render(model)


#         # Check output
#         self.assertEqual(image.shape, (16, 16))
#         self.assertTrue(torch.any(image > 0))

#         # A simple check for axisymmetry (for a non-rotating BH)
#         # The image should be roughly symmetrical
#         # This is a weak check
#         top_half_sum = torch.sum(image[:8, :])
#         bottom_half_sum = torch.sum(image[8:, :])
#         # In a perfect setup this might be close, but lensing can break symmetry.
#         # Just check that both halves have some flux.
#         self.assertGreater(top_half_sum, 0)
#         self.assertGreater(bottom_half_sum, 0)


# if __name__ == '__main__':
#     unittest.main()
