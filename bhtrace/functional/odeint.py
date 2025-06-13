import torch
import torch.jit as jit
from abc import ABC, abstractmethod

class ODEint(torch.nn.Module):
    '''
    Base class for ODE integration methods.
    '''

    def __init__(self):
        super().__init__()
        self.__native_dyn_dt__ = False
        self.outX = None
        self.LTE = None
        self.t_s = None

    def forward(self, term, X0, T=None, tspan=None, dt=0.15, nsteps=128, eps=1e-3, event_fn=None, step_fn=None, variable_step=False, reg=None):
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
        - step_fn: callable(t, X, LTE) - step-controlling function.

        ### Output format: 
        - sol: dict, keywords
            sol['X'] - coordinates of the solution, shape[nt, dim]
            sol['T'] - time grid of the solution(s)
            sol['LTE'] - local truncation error on each time step
        if tracking events:
            sol['event_T'] - time of the event (-1 if not achieved)
        '''
        self.__NullState__()

        # Determine stepping strategy
        if tspan is not None:
            self.t_s = tspan
            variable_step = False
        elif T is not None:
            self.t_s = torch.linspace(T[0], T[1], nsteps)
        elif variable_step and self.__native_dyn_dt__ is not None:
            self.step_control = self.__native_dyn_dt__
            self.t_s = torch.zeros(nsteps)
        elif step_fn is not None:
            self.step_control = step_fn
            variable_step = True
        else:
            raise NotImplementedError('Cannot establish task: time configuration not supported')

        # Update regularizing function:
        if reg is not None:
            self.reg = reg

        # Read event_fn
        if event_fn is not None:
            self.event_control = event_fn

        # Pre-definitions
        self.dim = X0.shape[0]
        self.outX = torch.zeros(nsteps, self.dim)
        self.LTE = torch.zeros(nsteps, self.dim)
        event_T = -1

        self.outX[0, :] = X0

        # Solving loop
        for nt in range(nsteps - 1):
            t, dt = self.step_control(nt)
            if self.event_control(t=t, XP=self.outX[nt, :]):
                self.outX[nt:, :] = self.outX[nt, :]
                event_T = self.t_s[nt]
                break
            self.outX[nt, :] = self.reg(t, self.outX[nt, :])
            self.outX[nt + 1, :], self.LTE[nt + 1, :] = self.__step__(term=term, t=t, X=self.outX[nt, :], dt=dt)

        sol = {'X': self.outX, 'T': self.t_s, 'LTE': self.LTE, 'event_T': event_T}

        return sol

    @abstractmethod
    def __step__(self, term, t, X, dt):
        '''
        Abstract method for performing a single integration step.

        ### Input:
        - term: callable(t, x) - equation function, representing x'
        - t: current time
        - X: current state
        - dt: time step

        ### Output:
        - X_: updated state
        - LTE: local truncation error
        '''
        pass

    def reg(self, t, X):
        '''
        Regularization function for the state.

        ### Input:
        - t: current time
        - X: current state

        ### Output:
        - X: regularized state
        '''
        return X

    def __NullState__(self):
        '''
        Restore solver initial state.
        '''
        self.event_control = self.__event_control__
        self.step_control = self.__step_control__

    def step_control(self, nt):
        '''
        Step control function.

        ### Input:
        - nt: current step index

        ### Output:
        - t: current time
        - dt: time step
        '''
        raise NotImplementedError('Step controller was not set during initialization')

    def __step_control__(self, nt):
        '''
        Default step control function.

        ### Input:
        - nt: current step index

        ### Output:
        - t: current time
        - dt: time step
        '''
        dt = self.t_s[nt + 1] - self.t_s[nt]
        return self.t_s[nt], dt

    def event_control(self, t, X):
        '''
        Event control function.

        ### Input:
        - t: current time
        - X: current state

        ### Output:
        - bool: whether the event condition is met
        '''
        return False

    def __event_control__(self, nt):
        '''
        Default event control function.

        ### Input:
        - nt: current step index

        ### Output:
        - bool: whether the event condition is met
        '''
        return False


class Euler(ODEint):
    '''
    Euler integration method.
    '''

    def __init__(self):
        super().__init__()

    def __step__(self, term, t, X, dt):

        f0 = term(t, X)
        X_ = X + f0 * dt
        t_ = t + dt

        # Local truncation error
        f1 = term(t_, X)
        f2 = term(t, X_)
        df_t = (f1 - f0) / dt
        df_x = (f2 - f0) / dt
        LTE = 0.5 * (dt ** 2) * (df_t + f0 * df_x)

        return X_, LTE


class RKF23b(ODEint):
    '''
    Runge-Kutta-Fehlberg 2(3) integration method.
    '''

    def __init__(self, N=5, varistep=False):
        super().__init__()
        self.C = torch.Tensor([0.0, 0.25, 27 / 40, 1.0])
        self.A = torch.Tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [-189 / 800, 729 / 800, 0.0, 0.0],
            [214 / 891, 1 / 33, 650 / 891, 0.0]
        ])
        self.B = torch.Tensor([214 / 891, 1 / 33, 650 / 891, 0.0])
        self.E = torch.Tensor([41 / 162, 0.0, 800 / 1053, -1 / 78])
        self.E = self.E - self.B
        self.h = 0.05

        self.max_dt = 1e-2
        self.min_dt = 1e-6

    def step_control(self, t, X, LTE):
        self.h = self.h * 0.9 * torch.pow(self.eps / (LTE + self.eps * 1e-2), 0.2)

    def eof_dyn_dt(self, t, X, Y, LTE, dt):
        t_, dt = self.__step_control__(t, X, LTE)
        if LTE < self.eps:
            X_ = X + Y @ self.B
            t_ = t + dt
            self.N = 10
            return t_, X_, LTE
        elif self.N > 0:
            self.N -= 1
            self.h = dt * 0.9 * torch.pow(self.eps / (LTE + self.eps * 1e-2), 0.2)
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
        Y[:, 0] = term(t, X)  # 1 step

        dX1 = self.A[1, 0] * Y[:, 0] * dt
        dT1 = self.C[1] * dt
        Y[:, 1] = term(t + dT1, X + dX1)  # 2 step

        dX2 = (self.A[2, 0] * Y[:, 0] + self.A[2, 1] * Y[:, 1]) * dt
        dT2 = self.C[2] * dt
        Y[:, 2] = term(t + dT2, X + dX2)  # 3 step

        dX3 = (self.A[3, 0] * Y[:, 0] + self.A[3, 1] * Y[:, 1] + self.A[3, 2] * Y[:, 2]) * dt
        dT3 = self.C[3] * dt
        Y[:, 3] = term(t + dT3, X + dX3)  # 4 step

        LTE = torch.linalg.vector_norm(Y @ self.E, ord=1)

        return self.eof(t, X, Y, LTE, dt)

class Verlet(ODEint):
    '''
    Verlet integration method.
    '''

    def __init__(self):
        super().__init__()

    def __step__(self, term, t, X, dt):
        X_half = X + 0.5 * term(t, X) * dt
        t_half = t + 0.5 * dt
        X_ = X + term(t_half, X_half) * dt
        LTE = torch.zeros_like(X)  # Verlet method does not provide LTE estimation
        return X_, LTE

ODE_SCHEMES = {'RKF23b': RKF23b, 'Euler': Euler, 'Verlet': Verlet}

def ODE(name='Euler', *args, **kwargs):
    '''
    Factory function to create ODE integrator instances.

    ### Input:
    - name: str - Name of the ODE integration method.
    - *args, **kwargs: Additional arguments for the ODE integrator.

    ### Output:
    - ODEint: Instance of the specified ODE integration method.
    '''
    if name in ODE_SCHEMES:
        return ODE_SCHEMES[name](*args, **kwargs)
    else:
        raise ValueError(f"ODE scheme {name} is not known")
