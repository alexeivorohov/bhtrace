import torch
import time
import pickle

import sys
import matplotlib.pyplot as plt
sys.path.append('.')

from bhtrace.geometry import SphericallySymmetric, Photon
from bhtrace.functional import cart2sph, sph2cart
from bhtrace.radiation import thin_disk, i2_r

