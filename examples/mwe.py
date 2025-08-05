import os
import sys
import inspect

import unittest
import torch

root_path = '/home/alexey/Work/bhtrace-dev'
sys.path.append(root_path)
sys.path.append(os.getcwd())

from bhtrace.geometry import KerrSchild, Photon, Observer
from bhtrace.fields import Maxwell
from bhtrace.tracing import PTracer

from bhtrace.grrt import ObjectAndDisk, ThinNewtonianDisk


# scene = Scene(ST, Photon, net, disk)

# IM1 = scene.compute(angle= ,saveas='', method=ptracer)


electrodynamics = Maxwell()
tracer = PTracer()
spacetime = KerrSchild()
photon = Photon(spacetime)
obs = Observer(spacetime=spacetime)
medium = ThinNewtonianDisk(spacetime=spacetime)
obs.setup_net()

scene = ObjectAndDisk('ObjectAndDisk', )

scene.propagate_photons()
