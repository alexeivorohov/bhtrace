'''
This module contains common methods for operating with arrays and functions


'''

from typing import Tuple
from itertools import permutations

import torch
import numpy as np


### Array utilities

def meshgrid4d(ts, rs, ths, phs):
    '''
    Constructs 4d meshgrid from given arrays

    Args:
    - ts: list of float - time coordinates
    - rs: list of float - radial coordinates
    - ths: list of float - theta coordinates
    - phs: list of float - phi coordinates

    Returns:
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

def net(shape='square', rng=(5, 5), YZ0=[0, 0], X0=20, YZsize=[8, 8]):
    '''
    Routine for generating coordinate grid on observer's sky

    Args:
    - shape: str - type of net to be generated [line, square, circle, hex]
    - rng: tuple - rank of a net
    - YZ0: list - initial YZ coordinates
    - X0: float - initial X coordinate
    - YZsize: list - size of the grid in Y and Z directions

    Returns:
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
        yy, zz = torch.meshgrid(smpl_y, smpl_z, indexing='ij')
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

def bisection(func: callable,
              x_min: float | torch.Tensor,
              x_max: float | torch.Tensor, 
              *args, 
              tol=1e-4,
              maxiter=100,
              **kwargs,
              ):
    '''
    Find function zeros by recursive bisection method.

    Args:
    - func: callable - function which zeros will be find,
    - x_min: torch.Tensor - lower boundary,
    - x_max: torch.Tensor - upper boundary,
    - *args: additional positional arguments to pass to the function,
    - tol: float - tolerance,
    - maxiter: int - maximum number of bisectiom iterations,
    - **kwargs: additional keyword arguments to pass to the function

    Returns:
    - outp: torch.Tensor - zeros found on the given interval
    '''
    if type(x_min) is float:
        x_min = torch.tensor([x_min])
    if type(x_max) is float:
        x_max = torch.tensor([x_max])

    mid = (x_min + x_max) / 2
    if maxiter == 0:
        return mid
    
    if args or kwargs:
        func_wrapped = lambda x: func(x, *args, **kwargs)
    
    f_mid = func(mid)
    tol_mask = torch.greater(abs(f_mid), tol)
    sign0 = torch.greater(func(x_min), 0)
    sign1 = torch.greater(func(x_max), 0)
    mask0 = torch.logical_and(torch.logical_xor(sign0, sign1), tol_mask)
    mid0 = mid[mask0]
    sign_mid = torch.greater(f_mid[mask0], 0)
    mask_mid = torch.logical_xor(sign0[mask0], sign_mid)
    not_mask_mid = torch.logical_not(mask_mid)
    new_xmin = x_min[mask0]
    new_xmax = x_max[mask0]
    new_xmax[mask_mid] = mid0[mask_mid]
    new_xmin[not_mask_mid] = mid0[not_mask_mid]
    res = bisection(func, new_xmin, new_xmax, *args, tol=tol, maxiter=maxiter-1, **kwargs)
    outp = mid
    outp[mask0] = res
    return outp

def find_domain(func, x_min, x_max, par, alpha=1.0, maxiter=10):
    '''
    Function to determine the function domain for a given function `func`.
    
    Args:
    - func: function to evaluate
    - x_min: torch.Tensor - minimum x values
    - x_max: torch.Tensor - maximum x values
    - par: torch.Tensor - parameters for the function
    - alpha: float - scaling factor
    - maxiter: int - maximum number of iterations
    
    Returns:
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


### Function utilities

def last_non_nan(X):
    '''
    For given array, 
    '''
    
    n = X.shape[0] - 1
    mask = torch.isnan(X)
    nonnan = 0
    for k in range(n):
        if mask[n-k] == False:
            nonnan = X[n-k]
            break
    return nonnan

def weightened_upsample_1d(
        X: torch.Tensor,
        tgt: torch.Tensor,
        func: callable = lambda tgt: abs(tgt),
        eps: float = 0.1,
        diff_threshold: float = 1e-1,
        diff_threshold_func: callable = None,
        mean_threshold_func: callable = None,
        fill: callable = lambda x, tgt: torch.nan
) -> Tuple[torch.Tensor]:
    '''
    Upsamples a 1D tensor `X` based on the values of a target tensor `tgt`.
    New points are inserted between existing points in `X` only if the change
    in `tgt` between those points is above a certain threshold.

    The position of the new points is weighted by the change in `tgt`. Where
    `tgt` changes a lot, the new point in `X` is closer to the point with
    the larger `tgt` value.

    If the absolute difference between consecutive `tgt` values is below
    `diff_threshold`, no new point is inserted for that segment.

    Args:
        X (torch.Tensor): The tensor to be upsampled along the first dimension. Shape (N, ...).
        tgt (torch.Tensor): The target tensor used to determine weights. Must be a 1D tensor of shape (N,) or (N, 1).
        func (callable, optional): A function to apply to `tgt` before
            calculating weights. Defaults to `abs(tgt)`.
        eps (float, optional): A small value added to weights to avoid
            division by zero. Defaults to 0.1.
        diff_threshold (float, optional): Threshold for `tgt` difference.
            If `abs(tgt[i+1] - tgt[i]) < diff_threshold`, no point is inserted.
            Defaults to 1e-2.
        diff_threshold_func (calllable, optional): Function of `tgt` difference to evaluate
            instead of condition described below. Must return boolean tensor of the same shape.
        mean_threshold_func (calllable, optional): Function of `tgt` mean to evaluate
            condition on means. Must return boolean tensor of the same shape.
        fill (callable, optional): A function to determine the value of `tgt`
            at the new points. Defaults to `lambda x, tgt: torch.nan`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - X_new (torch.Tensor): The upsampled `X` tensor. Shape (M, ...), where N <= M <= 2*N-1.
            - tgt_new (torch.Tensor): The `tgt` tensor with new points. Shape (M, ...).
            - mask (torch.Tensor): A boolean mask of shape (M,) indicating the positions of
              the newly inserted points.
    '''
    N = X.shape[0]
    if N <= 1:
        mask = torch.zeros(N, dtype=torch.bool, device=X.device)
        return X, tgt, mask

    # Create fully upsampled arrays of size 2N-1
    # w = (func(tgt) + eps).view(-1, 1)
    # xi = (w[1:] - w[:-1])/(w[1:] + w[:-1])
    # xi = torch.sigmoid(xi)
    dX = (X[1:, ...] - X[:-1, ...]) * 0.5

    x_shape = list(X.shape)
    x_shape[0] = 2 * N - 1
    X_full = torch.zeros(*x_shape, dtype=X.dtype, device=X.device)
    X_full[0::2, ...] = X
    X_full[1::2, ...] = X[:-1, ...] + dX

    tgt_shape = list(tgt.shape)
    tgt_shape[0] = 2 * N - 1
    tgt_full = torch.zeros(*tgt_shape, dtype=tgt.dtype, device=tgt.device)
    tgt_full[0::2, ...] = tgt
    tgt_full[1::2, ...] = fill(X, tgt)

    # Create a mask to select which points to keep
    # Original points are always kept
    keep_mask_orig = torch.zeros(2 * N - 1, dtype=torch.bool, device=X.device)
    keep_mask_orig[0::2] = True

    # New points are kept based on threshold
    tgt_diff = torch.abs(tgt[1:] - tgt[:-1])
    
    upsample_mask = torch.zeros_like(tgt_diff, dtype=torch.bool)
    if diff_threshold_func is None:
        upsample_mask.logical_or_(tgt_diff >= diff_threshold)
    else:
        upsample_mask.logical_or_(diff_threshold_func(tgt_diff))
    if mean_threshold_func is not None:
        tgt_mean = 0.5*(tgt[1:] + tgt[:-1])
        upsample_mask.logical_or_(mean_threshold_func(tgt_mean))

    # print(upsample_mask)
    keep_mask_new = torch.zeros(2 * N - 1, dtype=torch.bool, device=X.device)
    keep_mask_new[1::2] = upsample_mask.squeeze()

    keep_mask = torch.logical_or(keep_mask_orig, keep_mask_new)

    # Apply the mask to get the final arrays
    X_new = X_full[keep_mask]
    tgt_new = tgt_full[keep_mask]

    # The final mask indicates which points in the new array were inserted
    final_mask = keep_mask_new[keep_mask]

    return X_new, tgt_new, final_mask

def find_peaks(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds peaks in a 1D tabulated function by comparing with neighbor points.

    A point is considered a peak if its value is strictly greater than the
    values of its immediate neighbors. This method finds peaks that are
    located exactly on the grid points of the tabulated function.

    Args:
        x (torch.Tensor): The x-coordinates of the tabulated function (1D).
        y (torch.Tensor): The y-coordinates of the tabulated function (1D).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x_peaks (torch.Tensor): The x-coordinates of the found peaks.
            - y_peaks (torch.Tensor): The y-coordinates (heights) of the found peaks.
    """
    if x.shape[0] < 3 or y.shape[0] < 3:
        return torch.tensor([], device=x.device), torch.tensor([], device=y.device)

    # Find peaks by comparing with neighbors
    y_minus_1 = y[:-2]
    y_current = y[1:-1]
    y_plus_1 = y[2:]

    peak_mask = (y_current > y_minus_1) & (y_current > y_plus_1)
    
    # The indices are offset by 1 because we are looking at y[1:-1]
    peak_indices = torch.where(peak_mask)[0] + 1
    
    x_peaks = x[peak_indices]
    y_peaks = y[peak_indices]

    return x_peaks, y_peaks