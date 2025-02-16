import torch
import torch.linalg as LA
import time

import sys
sys.path.append('.')
from bhtrace.geometry import KerrSchild
from bhtrace.functional import points_generate

# Generating test points

ts = [0]
xs = [10, 3]
ys = [10]
zs = [0, -5]

X = points_generate(ts, xs, ys, zs)

# Setting up spacetime

ST = KerrSchild()

# Eye test

g = ST.g(X)

print(g[0, :, :])

print(torch.inverse(g[0]) @ g[0])

# ginv = ST.ginv(X)

# eye_test = g@ginv
# eye = torch.eye(4)

# for i in range(N_test_p):
#     print(torch.allclose(eye_test[i, :, :], eye))

