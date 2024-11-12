import torch

from abc import ABC, abstractmethod

# Features to implement now:
# - Solving on a fixed time grid and for fixed steps (for variable step methods)
# - Event tracking for both cases

# Constant (or grid) dt: - function calls just one time
# else - can be called multiple times at each step

ODE_SCHEMES = ['RKF23b', 'Euler']

class ODEint(torch.nn.Module):

    def __init__(self):

        self.__native_dyn_dt__ = False
        self.outX = None
        self.LTE = None
        self.t_s = None

        pass

    
    def forward(self, term, X0, T=None, tspan=None,\
        dt=0.15, nsteps=128, eps=1e-3, event_fn=None,\
        step_fn=None, variable_step=False):
        '''
        Solving ODE, defined by f(t, x) = x' for (X0, T0)

        ### Input:
        - term: callable(t, x) - equation function, representing x'
        - X0: initial conditions
        only one of the followed must be specified:
        - T: tuple (T0, T1), nsteps
        - tspan: time grid, default - None
   

        other options:
        - event_fn: callable(t, x) - event function, f <= 0 - integration stops
        - step_fn: callable(t, X, LTE) - step-controling function.

        ### Output format: 
        - sol: dict, keywords
            sol['X'] - coordinates of the solution, shape[nt, dim]
            sol['T'] - time grid of the solution(s)
            sol['LTE'] - local truncation error on each time step
        if tracking events:
            sol['event_T'] - time of the event (-1 if not acheived)
        '''
        self.__NullState__()

        # Determine stepping strategy
        if tspan != None:
            self.t_s = tspan
            variable_step == False
        elif T!= None:
            self.t_s = torch.linspace(T[0], T[1], nsteps)
        elif (variable_step == True) & (self.__native_dyn_dt__ != None) :
            self.step_control = self.__native_dyn_dt__
            self.t_s = torch.zeros(nsteps)
        elif step_fn != None:
            self.step_control = step_fn
            variable_step == True
        else:
            raise NotImplementedError('Can not establish task: time configuration not supported')

        # Read event_fn
        if event_fn == None:
            pass
        else:
            self.event_control = event_fn

        # Pre-definitions
        self.dim = X0.shape[0]
        self.outX = torch.zeros(nsteps, self.dim)
        self.LTE = torch.zeros(nsteps, self.dim)
        event_T = -1

        self.outX[0, :] = X0

        # Solving loop
        for nt in range(nsteps-1):
            t, dt = self.step_control(nt)
            if self.event_control(nt):
                event_T = self.t_s[nt]
                break
            self.outX[nt+1, :], self.LTE[nt+1, :] =\
            self.__step__(term=term, t=t, X=self.outX[nt, :], dt=dt)
                

        sol = {'X': self.outX, 'T': self.t_s, 'LTE': self.LTE, 'event_T': event_T}

        return sol

    @abstractmethod
    def __step__(self, term, t, X, dt):

        pass


    def __NullState__(self):
        '''
        Restore solver initial state
        '''
        self.event_control = self.__event_control__
        self.step_control = self.__step_control__

        pass


    def step_control(self, nt):

        raise NotImplementedError('Step controller was not set during initialization')


    def __step_control__(self, nt):

        dt = self.t_s[nt+1] - self.t_s[nt]
        #self.LTE
        #self.X

        return self.t_s[nt], dt


    def event_control(self, nt):

        return False


    def __event_control__(self, nt):

        return False


class Euler(ODEint):

    def __init__(self):

        pass


    def __step__(self, term, t, X, dt):

        f0 = term(t, X)
        X_ = X + f0*dt
        t_ = t + dt

        # local truncation error
        f1 = term(t_, X)
        f2 = term(t, X_)
        df_t = (f1 - f0)/dt
        df_x = (f2 - f0)/dt
        LTE = 0.5*(dt)**2 * (df_t + f0*df_x)

        return X_, LTE


class RKF23b(ODEint):

    def __init__(self, N=5, varistep=False):

        self.C = torch.Tensor([0.0, 0.25, 27/40, 1.0])
        self.A = torch.Tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0],
                [-189/800, 729/800, 0.0, 0.0],
                [214/891, 1/33, 650/891, 0.0]])
        self.B = torch.Tensor([214/891, 1/33, 650/891, 0.0])
        self.E = torch.Tensor([41/162, 0.0, 800/1053, -1/78])
        self.E = self.E - self.B
        self.h = 0.05

        self.max_dt = 1e-2
        self.min_dt = 1e-6
        
        # if varistep:
        #     self.eof = self.eof_dyn_dt
        # else:
        #     self.eof = self.eof_const_dt


    def step_control(self, t, X, LTE):

        self.h = dt*0.9*torch.pow(self.eps/(LTE+self.eps*1e-2), 0.2)


    def eof_dyn_dt(self, t, X, Y, LTE, dt):

        t_, dt = self.__step_control__
        if LTE < self.eps:
            X_ = X + Y @ self.B
            t_ = t + dt
            self.N = 10
            return t_, X_, LTE
        elif self.N > 0:
            self.N -= 1
            self.h = dt*0.9*torch.pow(self.eps/(LTE+self.eps*1e-2), 0.2)
            return self.__step__(term, t, X)
        else:
            return t, X, LTE


    def eof_const_dt(self, t, X, Y, LTE, dt):

        X_ = X + Y @ self.B
        t_ = t + dt

        return t_, X_, LTE


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

        LTE = torch.linalg.vector_norm(Y @ self.E, ord=1)
        
        return self.eof(t, X, Y, LTE, dt)
        

class Verlet(ODEint):

    def __init__(self):

        pass

    def __step__(self, term, t, X):

        

        pass
    



    


# class ODEX(ODEint):
