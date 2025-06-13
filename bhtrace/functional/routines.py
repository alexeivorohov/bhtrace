# This file contains some essential routines
import sys
import time
import torch
import numpy as np
from itertools import permutations

def cart2sph(inX, inP):
    '''
    Cast coordinates and impulses 4-d from cartesian to 4-d spherical coordinates.

    Only contravariant representation!

    ### Inputs:
    - inX: torch.Tensor - input coordinates 
    - inP: torch.Tensor - input impulses (or velocities)

    ### Outputs:
    - tuple(outX, outP): torch.Tensor - output coordinates and impulses in spherical coordinates
    '''
    shape = inX.shape
    X = inX.view(-1, 4)
    P = inP.view(-1, 4)

    T0, X0, Y0, Z0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    vT, vX, vY, vZ = P[:, 0], P[:, 1], P[:, 2], P[:, 3]

    X2Y2 = X0**2 + Y0**2
    R = torch.sqrt(X2Y2 + Z0**2)
    TH = torch.arccos(Z0/R)
    PH = torch.arctan2(Y0, X0)

    vR = torch.sin(TH)*(torch.cos(PH)*vX + torch.sin(PH)*vY) + torch.cos(TH)*vZ
    vTH = (torch.cos(TH)*(torch.cos(PH)*vX+torch.sin(PH)*vY) - torch.sin(TH)*vZ)/R
    vPH = (-torch.sin(PH)*vX+torch.cos(PH)*vY)/R*torch.sin(TH)

    outX = torch.zeros_like(X)
    outP = torch.zeros_like(P)

    outX[:, 0], outX[:, 1], outX[:, 2], outX[:, 3] = T0, R, TH, PH
    outP[:, 0], outP[:, 1], outP[:, 2], outP[:, 3] = vT, vR, vTH, vPH

    return outX.view(shape), outP.view(shape)


def sph2cart(inX, inP):
    '''
    Cast 4d spherical coordinates to 4d cartesian coordinates.

    Only contravariant representation!

    ### Inputs:
    - inX: torch.Tensor - input coordinates 
    - inP: torch.Tensor - input impulses (or velocities)

    ### Outputs:
    - tuple(outX, outP): torch.Tensor - output coordinates and impulses
    '''
    shape = inX.shape
    X = inX.view(-1, 4)
    P = inP.view(-1, 4)

    T, R, TH, PH = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    vT, vR, vTH, vPH = P[:, 0], P[:, 1], P[:, 2], P[:, 3]

    outX = torch.zeros_like(X)
    outP = torch.zeros_like(P)

    outX[:, 0] = T
    outX[:, 1] = R*torch.sin(TH)*torch.cos(PH)
    outX[:, 2] = R*torch.sin(TH)*torch.sin(PH)
    outX[:, 3] = R*torch.cos(TH)

    outP[:, 0] = vT
    outP[:, 1] = (vR*torch.sin(TH) + R*torch.cos(TH)*vTH)*torch.cos(PH) - R*torch.sin(PH)*torch.sin(TH)*vPH
    outP[:, 2] = (vR*torch.sin(TH) + R*torch.cos(TH)*vTH)*torch.sin(PH) + R*torch.cos(PH)*torch.sin(TH)*vPH
    outP[:, 3] = vR*torch.cos(TH) - R*torch.sin(TH)*vTH

    return outX.view(shape), outP.view(shape)


def points_generate(ts, rs, ths, phs):
    '''
    Make all permutations for 4 coordinates lists.

    ### Inputs:
    - ts: list of float - time coordinates
    - rs: list of float - radial coordinates
    - ths: list of float - theta coordinates
    - phs: list of float - phi coordinates

    ### Outputs:
    - X: torch.Tensor - tensor of shape (len(ts)*len(rs)*len(ths)*len(phs), 4) containing all permutations
    '''
    N_test_p = len(ts)*len(rs)*len(ths)*len(phs)
    X = torch.zeros(N_test_p, 4)

    i = 0
    for t in ts:
        for r in rs:
            for th in ths:
                for ph in phs:
                    X[i, :] = torch.Tensor([t, r, th, ph])
                    i += 1

    return X


def EulerRotation(
        X: torch.Tensor,
        dphi: float | torch.Tensor,
        dth: float | torch.Tensor
        ):
    '''
    Euler rotation of unit vector(s) in cartesian coordinates

    Inputs:
    - X: torch.Tensor - vector(s) 
    - dphi: float | torch.Tensor - azimuthal rotation
    - dth: float | torch.Tensor - ... rotation

    Outputs:
    - outX: torch.Tensor - rotated vector
    '''
    outX = X.copy()

    cdphi = torch.cos(dphi)
    sdphi = torch.sin(dphi)

    sdth = torch.sin(dth)
    cdth = torch.cos(dth)

    outX[..., 0] = (X[..., 0]*cdphi - X[..., 1]*sdphi)*cdth - X[..., 0]*X[..., 2]*sdth
    outX[..., 1] = (X[..., 0]*sdphi + X[..., 1]*cdphi)*cdth - X[..., 1]*X[..., 2]*sdth
    outX[..., 2] = torch.pow((X[..., 0]**2 + X[..., 1]**2), -0.5)*sdth + X[..., 2]*cdth

    return outX


def net(shape='square', rng=(5, 5), YZ0=[0, 0], X0=20, YZsize=[8, 8]):
    '''
    Routine for generating coordinate grid on observer's sky

    ### Inputs:
    - shape: str - type of net to be generated [line, square, circle, hex]
    - rng: tuple - rank of a net
    - YZ0: list - initial YZ coordinates
    - X0: float - initial X coordinate
    - YZsize: list - size of the grid in Y and Z directions

    ### Outputs:
    - tuple(xx, yy, zz): torch.Tensor - coordinates of the grid
    '''
    yy = []
    zz = []

    if shape == 'line':
        yy = torch.linspace(-0.5, 0.5, rng[0]).view(-1, 1)
        zz = torch.linspace(-0.5, 0.5, rng[0]).view(-1, 1)
    elif shape == 'square':
        smpl_y = torch.linspace(-0.5, 0.5, rng[0])
        smpl_z = torch.linspace(-0.5, 0.5, rng[1])
        yy, zz = torch.meshgrid(smpl_y, smpl_z)
    elif shape == 'circle':
        ph = [torch.linspace(0.0, 2.0, 4*(n+1)+1)[1:] for n in range(rng[0]-1)]
        ph = torch.cat(ph)*torch.pi
        r = [torch.ones(4*(n+1)+1)[1:]*(n+1)/rng[0] for n in range(rng[0]-1)]
        r = torch.cat(r)
        yy = r*torch.sin(ph)
        zz = r*torch.cos(ph)
    elif shape == 'hex':
        raise NotImplementedError('hex shape is not implemented')

    xx = torch.ones_like(yy)*X0
    yy = yy*YZsize[0] + YZ0[0]
    zz = zz*YZsize[1] + YZ0[1]

    return xx.flatten(), yy.flatten(), zz.flatten()


def bisection(func: callable, x_min: torch.Tensor, x_max: torch.Tensor, par: torch.Tensor, tol=1e-4, maxiter=100):
    '''
    Find function zeros by recursive bisection method.

    ### Inputs:
    - func: callable - function to find zeros
    - x_min: torch.Tensor - lower boundary
    - x_max: torch.Tensor - upper boundary
    - par: torch.Tensor - parameters for the function
    - tol: float - tolerance for zero finding
    - maxiter: int - maximum number of iterations

    ### Outputs:
    - outp: torch.Tensor - zeros found on the given interval
    '''
    mid = (x_min + x_max) / 2

    if maxiter == 0:
        return mid
    else:
        f_mid = func(mid, par)
        tol_mask = torch.greater(abs(f_mid), tol)
        sign0 = torch.greater(func(x_min, par), 0)
        sign1 = torch.greater(func(x_max, par), 0)
        mask0 = torch.logical_and(torch.logical_xor(sign0, sign1), tol_mask)
        mid0 = mid[mask0]
        sign_mid = torch.greater(f_mid[mask0], 0)
        mask_mid = torch.logical_xor(sign0[mask0], sign_mid)
        not_mask_mid = torch.logical_not(mask_mid)
        new_xmin = x_min[mask0]
        new_xmax = x_max[mask0]
        new_xmax[mask_mid] = mid0[mask_mid]
        new_xmin[not_mask_mid] = mid0[not_mask_mid]
        res = bisection(func, new_xmin, new_xmax, par[mask0], tol=tol, maxiter=maxiter-1)
        outp = mid
        outp[mask0] = res
        return outp


def def_fspace(func, x_min, x_max, par, alpha=1.0, maxiter=10):
    '''
    Function to define the function space for a given function `func`.
    
    ## Inputs:
    - func: function to evaluate
    - x_min: torch.Tensor - minimum x values
    - x_max: torch.Tensor - maximum x values
    - par: torch.Tensor - parameters for the function
    - alpha: float - scaling factor
    - maxiter: int - maximum number of iterations
    
    ## Outputs:
    - x_min: torch.Tensor - updated minimum x values
    - x_max: torch.Tensor - updated maximum x values
    '''
    for _ in range(maxiter):
        new_xmax = x_max + alpha * (x_max - x_min)
        f1 = func(new_xmax, par)
        xmask1 = ~torch.isnan(f1)
        if torch.all(xmask1):
            return x_min, new_xmax
        x_max[xmask1] = new_xmax[xmask1]
    return x_min, x_max


def levi_civita_tensor(dim):
    '''
    Basic construction of fully-antisymmetric Levi-Civita object in dimension=dim.

    Not invariant form!

    ### Inputs:
    - dim: int - Dimension of the tensor.

    ### Outputs:
    - outp: torch.Tensor - Levi-Civita tensor of shape (dim, dim, ..., dim).
    '''
    outp = torch.zeros((dim,) * dim, dtype=torch.int8)  # Create a dim-dimensional tensor filled with zeros

    # Generate all permutations of dimensions
    for perm in permutations(range(dim)):
        # Calculate the sign of the permutation
        sign = 1
        for i in range(dim):
            for j in range(i + 1, dim):
                if perm[i] > perm[j]:
                    sign *= -1
        outp[perm] = sign  # Assign the sign to the appropriate position in the tensor

    return outp


def print_status_bar(progress, total, elapsed_time):
    '''
    This function displays and updates status bar.

    ### Inputs:
    - progress: int - current progress
    - total: int - total steps
    - elapsed_time: float - elapsed time in seconds
    '''
    bar_length = 20  # Length of the status bar
    filled_length = int(bar_length * progress // total)  # Calculate filled length
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)  # Create the bar
    percentage = (progress / total) * 100  # Calculate percentage
    status = f'\r {percentage:.2f}% [{bar}] | '

    # Estimate remaining time
    if progress == 0:
        info = "N/A tr/s | T: N/A s | ETA N/A s"
    elif progress == total:  
        trs = progress / elapsed_time
        info = f'{trs:.2f} tr/s | T: {elapsed_time:.2f} s | Done !' 
    else:
        trs = progress / elapsed_time
        ETA_t = (total - progress) / trs
        info = f'{trs:.2f} tr/s | T: {elapsed_time:.2f} s | ETA {ETA_t:.2f} s' 

    sys.stdout.write(status + info)
    sys.stdout.flush()
