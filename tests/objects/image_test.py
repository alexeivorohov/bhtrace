import torch
import time

import sys
sys.path.append('.')
from bhtrace.geometry import EffGeomSPH, Photon
from bhtrace.electrodynamics import Maxwell, EulerHeisenberg
from bhtrace.scenarios import ImagePix

q = 0.6

ED0 = Maxwell()
ED1 = EulerHeisenberg(h=1)
E = lambda X: torch.Tensor([0, q/X[1], 0, 0])
B = lambda X: torch.zeros(4)

f = lambda r: 1.0 - 2.0/r
f_r = lambda r: 2.0 * torch.Tensor(r, -2)
ST0 = EffGeomSPH(ED=ED0, f=f, f_r=f_r, E=E, B=B)
phot0 = Photon(ST0)

ImagePix(
    particle = phot0,
    rad_field = None,
    p_width = 16,
    p_height = 16,
    pixel_d = 0.1,
    nsteps = 128,
    T = 20.0,
    save_as = 'try.pkl'
    )





