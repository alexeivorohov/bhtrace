import torch
import time

import sys
sys.path.append('.')
from bhtrace.geometry import MinkowskiSph
from bhtrace.electrodynamics import Maxwell

ED = Maxwell()
