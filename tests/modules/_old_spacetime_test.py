import torch
import time

import sys
sys.path.append('.')
from bhtrace.geometry import Spacetime, SphericallySymmetric, MinkowskiSph

# Generating test points

ts = [0]
rs = [5]
ths = [0.1, 3]
phs = [3]
N_test_p = len(ts)*len(rs)*len(ths)*len(phs)

X = torch.zeros(N_test_p, 4)

i = 0
for t in ts:
    for r in rs:
        for th in ths:
            for ph in phs:
                X[i, :] = torch.Tensor([t, r, th, ph])
                i += 1


# Setting up Schwarzshild spacetime

schw = lambda r: 1 - 2/r
schw_r = lambda r: 2*torch.pow(r, -2)

ST = SphericallySymmetric(A=schw, f_r=schw_r)
# ST = MinkowskiSph()

# Eye test

g = ST.g(X)
ginv = ST.ginv(X)

eye_test = g@ginv
eye = torch.eye(4)

for i in range(N_test_p):
    print(torch.allclose(eye_test[i, :, :], eye))

# Tetrad test?


# dg test

FlatST = SphericallySymmetric(A=lambda r: 1, f_r = lambda r: 0)
dgX = FlatST.dg(X)

print(dgX)
print(dgX.shape)


# Seems correct
dgX = ST.dg(X)
print(dgX)
print(dgX.shape)

# Connections

t0 = time.time()
gma = ST.conn(X)
t1 = time.time()
gma_ = ST.conn_(X)
t2 = time.time()

print(t1-t0)
print(t2-t1)

