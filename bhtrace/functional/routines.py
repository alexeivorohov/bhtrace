# This file contains some essential routines

import torch

# Cooridnate transformations, OK
def cart2sph(inX, inP):

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

    shape = inX.shape
    X = inX.view(-1, 4)
    P = inP.view(-1, 4)

    T, R, TH, PH,  = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    vT, vR, vTH, vPH = P[:, 0], P[:, 1], P[:, 2], P[:, 3]

    outX = torch.zeros_like(X)
    outP = torch.zeros_like(P)

    outX[:, 0] = T
    outX[:, 1] = R*torch.sin(TH)*torch.cos(PH)
    outX[:, 2] = R*torch.sin(TH)*torch.sin(PH)
    outX[:, 3] = R*torch.cos(TH)

    outP[:, 0] = vT
    outP[:, 1] = (vR*torch.sin(TH) + R*torch.cos(TH)*vTH)*torch.cos(PH) \
        - R*torch.sin(PH)*torch.sin(TH)*vPH
    outP[:, 2] = (vR*torch.sin(TH) + R*torch.cos(TH)*vTH)*torch.sin(PH) \
        + R*torch.cos(PH)*torch.sin(TH)*vPH
    outP[:, 3] = vR*torch.cos(TH)-R*torch.sin(TH)*vTH

    return outX.view(shape), outP.view(shape)

# Points generating:

def points_generate(ts, rs, ths, phs):

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

# Should be remaked or removed
def net(type='square', rng=5, db=[-5,5,-5,5], D0 = 20, dth=0, dph=0):
    '''
    Function for generation of initial photons grid on observer's sky
    type - type of net to be generated [line, square, circle]
    rng - rank of a net, 
    db -
    D0 -
    dth -
    dph 
    '''

    # Coordinates
    yy = []
    zz = []

    if type == 'line':

        xx = D0*torch.ones([rng, 1])
        yy = torch.linspace(db[0], db[1], rng).view(-1, 1)
        zz = db[2]*torch.ones([rng, 1])

    if type == 'square':

        xx = D0*torch.ones([rng, rng])
        smpl_y = torch.linspace(db[0], db[1], rng)
        smpl_z = torch.linspace(db[2], db[3], rng)
        yy, zz = torch.meshgrid(smpl_y, smpl_z)

    if type == 'circle':

        ph = [torch.linspace(0, 2*np.pi, 4*(n+1)) for n in range(rng-1)]
        ph = torch.cat(ph)
        r = [torch.ones(4*(k+1))*(k+1)/rng for k in range(rng-1)]
        r = torch.cat(r)*abs(db[0])

        xx = D0*torch.ones_like(r)
        yy = r*torch.sin(ph)
        zz = r*torch.cos(ph)

    if type == 'hex':
      pass

    # Velocities
    vx = - torch.ones_like(xx.view(-1,1))
    vy = torch.zeros_like(vx)
    vz = torch.zeros_like(vx)

    if dth != 0:
        x_ = xx*np.cos(dth) - zz*np.sin(dth)
        zz = xx*np.sin(dth) + zz*np.cos(dth)
        xx = x_

        vz = vx*np.sin(dth)
        vx = vx*np.cos(dth)

    if dph != 0:
        x_ = xx*np.cos(dph) - yy*np.sin(dph)
        yy = xx*np.sin(dph) + yy*np.cos(dph)
        xx = x_

        vy = vx*np.sin(dph)
        vx = vx*np.cos(dph)

    print(xx.shape)
    print(yy.shape)
    print(zz.shape)

    return xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1), vx, vy, vz

# Works?
def bisection(func: callable, x_min: torch.Tensor, x_max: torch.Tensor, par: torch.Tensor, tol=1e-4, maxiter=100):
    '''
    eq: callable X
    intervals: intervals[0] must be x_min, intervals[1] must be x_max
    '''
    mid = (x_min+x_max)/2

    if maxiter==0:

        return mid

    else:
        
        f_mid = func(mid, par)

        tol_mask = torch.greater(abs(f_mid), tol)
        
        sign0 = torch.greater(func(x_min, par), 0)
        sign1 = torch.greater(func(x_max, par), 0)

        # mask0 = torch.logical_xor(sign0, sign1)

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

# Works?
def def_fspace(func: callable, x_min: torch.Tensor, x_max: torch.Tensor, par: torch.Tensor, alpha=0.5, maxiter=100):
    '''
    eq: callable X
    intervals: intervals[0] must be x_min, intervals[1] must be x_max
    '''

    if maxiter==0:

        return x_min, x_max

    else:
        
        d = (x_min+x_max)*alpha
        
        new_xmax = x_max+d

        f1 = func(new_xmax, par)

        mask1 = torch.isnan(f1)

        xmask1 = torch.logical_not(mask1)

        _, res1 = def_fspace(func, x_min[mask1], x_max[mask1], par[mask1], alpha=alpha*0.7, maxiter=maxiter-1)

        _, resX = def_fspace(func, x_min[xmask1], new_xmax[xmask1], par[xmask1], alpha=alpha*1.3, maxiter=maxiter-1)

        x_max[mask1] = res1
        x_max[xmask1] = resX

        return x_min, x_max


def levi_civita_tensor(dim):
    # Create a tensor to hold the Levi-Civita symbol
    outp = torch.zeros((dim,) * dim)  # Create a dim-dimensional tensor filled with zeros
    
    # Generate all permutations of dimensions
    for perm in torch.permutations(torch.arange(dim)):
        # Calculate the sign of the permutation
        sign = 1 if perm[0].item() < perm[1].item() else -1
        for i in range(dim):
            for j in range(i + 1, dim):
                if perm[i] > perm[j]:
                    sign *= -1
        arr[tuple(perm)] = sign  # Assign the sign to the appropriate position in the tensor
    
    return outp