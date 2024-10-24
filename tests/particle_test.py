import torch

import sys
sys.path.append('.')
from bhtrace import SphericallySymmetric, Photon

# Generating test points

ts = [0]
rs = [2.2, 200]
ths = [0.1, 1]
phs = [0, 3, 6]
N_test_p = len(ts)*len(rs)*len(ths)*len(phs)

X = torch.zeros(N_test_p, 4)

i = 0
for t in ts:
    for r in rs:
        for th in ths:
            for ph in phs:
                X[i, :] = torch.Tensor([t, r, th, ph])
                i += 1

# Setting up SchwST and photon:

schw = lambda r: 1 - 2/r
schw_r = lambda r: 2*torch.pow(r, -2)

SchwST = SphericallySymmetric(f=schw, f_r=schw_r)

Phot = Photon(SchwST)

# Hamiltonian test

p0 = [0.6, 0.8, 0, 0]

P = torch.Tensor([p0 for i in range(N_test_p)])

print(P.shape)

print(X.shape)

H = Phot.Hmlt(X, P)

print(H.shape)

pre = torch.ones_like(X)
DVec = torch.einsum('bi,ij->bij', pre, torch.eye(4))

dH = Phot.dHmlt(X, P, DVec, 2e-5)

print(dH)



