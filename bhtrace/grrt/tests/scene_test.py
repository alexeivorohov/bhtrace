import os
import sys
import inspect

import unittest
import torch

root_path = '/home/alexey/Work/bhtrace-dev'
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.geometry import KerrSchild, Photon, Observer
from bhtrace.tracing import PTracer
from bhtrace.grrt import Scene

class TestScenes(unittest.TestCase):

    def test_ObjectAndDisk(self):

        # Constants and definitions
    
        tracer = PTracer()
        spacetime = KerrSchild()
        photon = Photon(spacetime)
        obs = Observer(spacetime=spacetime)
        obs.setup_net()

        scene = Scene('ObjectAndDisk')
        
        scene.propagate_photons()

        scene.compute_radiation()


if __name__ == '__main__':

    unittest.main()
