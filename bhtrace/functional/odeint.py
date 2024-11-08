import torch

from abc import ABC, abstractmethod

class ODEint(torch.nn.Module):

    def __init__(self):

        self.eps = 1e-3
        pass

    @abstractmethod
    def forward(self, term, X0, T0, dt):

        pass

    def set_control(self, control_fn):

        self.control_fn = control_fn
        pass



class RKF23b(ODEint):

    def __init__(self, Nl=2):

        self.Nl = Nl
        
        self.C = torch.Tensor([0.0, 0.25, 27/40, 1.0])
        self.A = torch.Tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0],
                [-189/800, 729/800, 0.0, 0.0],
                [214/891, 1/33, 650/891, 0.0]])
        self.B = torch.Tensor([214/891, 1/33, 650/891, 0.0])
        self.E = torch.Tensor([41/162, 0.0, 800/1053, -1/78])
        self.E = self.E - self.B
        self.h = 0.15
        pass


    def dt(self, err, dt):

        return dt*0.9*torch.pow(self.eps/err, 0.2)


    def __step__(self, term, t, X):

        Y = torch.zeros(X.shape[0], 4)
        dt = self.h
        Y[:, 0] = term(t, X) #1 step

        dX1 = self.A[1, 0]*Y[:, 0]*dt 
        dT1 = self.C[1]*dt
        Y[:, 1] = term(t+dT1, X+dX1) # 2 step

        dX2 = (self.A[2, 0]*Y[:, 0] + self.A[2, 1]*Y[:, 1])*dt
        dT2 = self.C[2]*dt
        Y[:, 2] = term(t+dT2, X+dX2) # 3 step

        dX3 = (self.A[3, 0]*Y[:, 0] + self.A[3, 1]*Y[:, 1] + self.A[3, 2]*Y[:, 2])*dt
        dT3 = self.C[3]*dt
        Y[:, 3] = term(t+dT3, X+dX3) # 4 step

        err = sum(abs(Y @ self.E))
        self.h = self.dt(err, dt)

        if err < self.eps:
            X_ = X + Y @ self.B
            t_ = t + dt
            return t_, X_, err
        else:
            return self.__step__(term, t, X)


    def forward(self, term, T0, X0, dt=0.15, nsteps=128, eps=1e-3):
        '''
        Solving ODE, defined by f(t, x) = x' for (X0, T0)

        ### Input:
        - term: callable(t, x) - equation function, representing x'
        - X0:
        - T0: 
        - dt: float
        - nsteps:int

        ### Output: 
        - sol: dict, keywords - 'X', 'T', 'err'
        - sol['X'] - coordinates of the solution
        - sol['T'] - time grid of the solution
        - sol['err'] - estimated errors on each time step
        '''

        self.eps = eps
        outX = torch.zeros(nsteps, X0.shape[0])
        t_s = torch.zeros(nsteps)
        err_s = torch.zeros(nsteps)

        outX[0, :] = X0
        t_s[0] = T0
        
        for nt in range(nsteps-1):
            t_s[nt+1], outX[nt+1, :], err_s[nt+1] = self.__step__(
                term, t_s[nt], outX[nt, :])
            
        sol = {'X': outX, 'T': t_s, 'err': err_s}

        return sol



    


# class ODEX(ODEint):
