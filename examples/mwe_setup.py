import torch
import time
import os

import sys
sys.path.append('/home/alexey/Work/bhtrace-dev/')

import torch

from bhtrace.geometry import Photon, MinkowskiCart, KerrSchild, SchwSchild


Ni = 25

spacetimes = {
    # 'Minkowski': MinkowskiCart(),
    'SchwSchild' : SchwSchild()
    # 'KerrSchild': KerrSchild(a=0.2, m=1, Q=0),
    }