'''
This file describes procedures of numerical differentiation, used by the library

'''
import torch
from abc import ABC, abstractmethod

class Diff(ABC):
    '''
    Abstract base class for numerical differentiation
    '''

    def __init__(self, func: callable):
        self.func = func

    @abstractmethod
    def __call__(self, X, params=None):
        pass


class Grad(Diff):
    '''
    Class for calculating the gradient of a function
    '''

    def __init__(self, func: callable, eps=1e-5):
        super().__init__(func)
        self.eps = eps

    def __call__(self, X, params=None):
        grad = torch.zeros_like(X)
        dVec = torch.eye(X.shape[0]) * self.eps

        for i in range(X.shape[0]):
            if params is not None:
                grad[i] = (self.func(X + dVec[i], params) - self.func(X - dVec[i], params)) / (2 * self.eps)
            else:
                grad[i] = (self.func(X + dVec[i]) - self.func(X - dVec[i])) / (2 * self.eps)

        return grad


class Hessian(Diff):
    '''
    Class for calculating the Hessian matrix of a function
    '''

    def __init__(self, func: callable, eps=1e-5):
        super().__init__(func)
        self.eps = eps

    def __call__(self, X, params=None):
        hessian = torch.zeros(X.shape[0], X.shape[0])
        dVec = torch.eye(X.shape[0]) * self.eps

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if params is not None:
                    f_ij = self.func(X + dVec[i] + dVec[j], params)
                    f_i_j = self.func(X + dVec[i] - dVec[j], params)
                    f_ij_ = self.func(X - dVec[i] + dVec[j], params)
                    f_i_j_ = self.func(X - dVec[i] - dVec[j], params)
                else:
                    f_ij = self.func(X + dVec[i] + dVec[j])
                    f_i_j = self.func(X + dVec[i] - dVec[j])
                    f_ij_ = self.func(X - dVec[i] + dVec[j])
                    f_i_j_ = self.func(X - dVec[i] - dVec[j])
                hessian[i, j] = (f_ij - f_i_j - f_ij_ + f_i_j_) / (4 * self.eps ** 2)

        return hessian


class DiffLinear(Diff):
    '''
    Class for calculating the first-order numerical derivative of a function
    '''

    def __init__(self, func: callable, eps=1e-5):
        super().__init__(func)
        self.eps = eps

    def __call__(self, X, params=None):
        dX = torch.eye(X.shape[0]) * self.eps
        diff = torch.zeros_like(X)

        for i in range(X.shape[0]):
            if params is not None:
                diff[i] = (self.func(X + dX[i], params) - self.func(X - dX[i], params)) / (2 * self.eps)
            else:
                diff[i] = (self.func(X + dX[i]) - self.func(X - dX[i])) / (2 * self.eps)

        return diff


class DiffHOrder(Diff):
    '''
    Class for calculating higher-order numerical derivatives of a function
    '''

    def __init__(self, func: callable, eps=1e-5, order=2):
        super().__init__(func)
        self.eps = eps
        self.order = order

    def __call__(self, X, params=None):
        dX = torch.eye(X.shape[0]) * self.eps
        diff = torch.zeros_like(X)

        if self.order == 2:
            for i in range(X.shape[0]):
                if params is not None:
                    diff[i] = (self.func(X + dX[i], params) - self.func(X - dX[i], params)) / (2 * self.eps)
                else:
                    diff[i] = (self.func(X + dX[i]) - self.func(X - dX[i])) / (2 * self.eps)

        elif self.order == 4:
            for i in range(X.shape[0]):
                if params is not None:
                    diff[i] = (-1 / 12 * self.func(X + 2 * dX[i], params) +
                               2 / 3 * self.func(X + dX[i], params) -
                               2 / 3 * self.func(X - dX[i], params) +
                               1 / 12 * self.func(X - 2 * dX[i], params)) / self.eps
                else:
                    diff[i] = (-1 / 12 * self.func(X + 2 * dX[i]) +
                               2 / 3 * self.func(X + dX[i]) -
                               2 / 3 * self.func(X - dX[i]) +
                               1 / 12 * self.func(X - 2 * dX[i])) / self.eps

        else:
            raise ValueError("The order value is not valid")

        return diff


