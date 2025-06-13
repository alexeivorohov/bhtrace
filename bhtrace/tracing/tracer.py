import torch
import os
import pickle
import time
from abc import abstractmethod
from ..geometry import Spacetime, Particle
from ..functional import ODE, print_status_bar


class Tracer():

    '''
    Base class for tracing particle trajectories in a given spacetime.
    '''

    def __init__(self, ode_method='Euler'):
        '''
        Initialize the Tracer with a specified ODE integration method.

        Parameters:
        - ode_method: str - The ODE integration method to use (default: 'Euler').
        '''
        self.odeint = ODE(ode_method)

    @abstractmethod
    def __term__(self, t, XP):
        '''
        Abstract method to define the term for ODE integration.

        Parameters:
        - t: float - The current time.
        - XP: torch.Tensor - The current state vector (position and momentum).

        Returns:
        - torch.Tensor: The term for ODE integration.
        '''
        pass

    def evnt(self, t, XP):

        # cr1 = self.particle.crit(XP[:4], XP[4:])
        cr1 = torch.less(self.max_proper_t, XP[0])
        cr2 = torch.less(self.r_max, torch.abs(XP[1]))
        # integration continues while function returns false
        return cr1 + cr2

    def forward(self,
                particle: Particle,
                X0,
                P0,
                T,
                nsteps=128,
                r_max=30.0,
                max_proper_t=500.0,
                dev='cpu',
                eps=1e-3):
        '''
        Trace particle trajectories.

        Parameters:
        - particle: Particle - The particle to trace.
        - X0: torch.Tensor [n, 4] - Initial positions.
        - P0: torch.Tensor [n, 4] - Initial impulses.
        - T: float - Simulation time on observer's clock.
        - nsteps: int - Number of steps per trajectory (default: 128).
        - r_max: float - Distance from center, on which computation should be stopped (default: 30.0).
        - max_proper_t: float - Particle proper time, on which computation should be stopped (default: 500.0).
        - dev: str - Device to perform computation (default: 'cpu').
        - eps: float - Parameter for numerical differentiation (default: 1e-3).

        Returns:
        - tuple: (X, P) where X is the positions and P is the impulses.
        '''

        self.particle = particle
        self.spc = particle.spacetime

        self.Nt = nsteps
        self.Ni = X0.shape[0]
        self.eps = eps

        self.r_max = torch.tensor([r_max], device=dev)
        self.max_proper_t = torch.tensor([max_proper_t], device=dev)

        self.X = torch.zeros(nsteps, self.Ni, 4, device=dev)
        self.P = torch.zeros(nsteps, self.Ni, 4, device=dev)
        self.X[0, :, :] = X0
        self.P[0, :, :] = P0

        T0 = torch.tensor([0.0], device=dev)

        start_time = time.time()

        print_status_bar(0, self.Ni, 0)

        for n in range(self.Ni):
            XP0 = torch.cat((X0[n], P0[n]))

            sol = self.odeint.forward(
                term=self.__term__,
                X0=XP0,
                T=(0.0, T),
                nsteps=nsteps,
                event_fn=self.evnt,
                reg=self.reg
            )

            self.X[:, n, :] = sol['X'][:, :4]
            self.P[:, n, :] = sol['X'][:, 4:]
            elapsed_time = time.time() - start_time
            print_status_bar(n + 1, self.Ni, elapsed_time)
        
        print('')

        return self.X, self.P


    def save(self, filename, directory=None, comment=None):
        '''
        Save the last result to a file specified by filename and optional directory.

        Parameters:
        - filename: str - The name of the file or full path.
        - directory: str, optional - Directory path to save the file in.
        - comment: str, optional - Additional comment to save with the result.

        Returns:
        - str: The full path of the saved file.
        '''

        if directory is not None:
            full_path = os.path.join(directory, filename)
        else:
            full_path = filename

        result = {
            # 'particle': self.particle,
            'trcr': self.name,
            'Ni': self.Ni,
            'X': self.X,
            'P': self.P
        }

        if comment is not None:
            result['comment'] = comment

        with open(full_path, 'wb') as file:
            pickle.dump(result, file)

        print(f'Results saved at {full_path}')

        return full_path

class MockTracer(Tracer):
    
    def __init__(self, particle, spacetime, ode_method='Euler'):
        '''
        Initialize the MockTracer with a specified particle and spacetime.

        Parameters:
        - particle: Particle - The particle to trace.
        - spacetime: Spacetime - The spacetime in which the particle moves.
        - ode_method: str - The ODE integration method to use (default: 'Euler').
        '''

        self.particle = particle
        self.spc = spacetime
        self.name = self.__class__.__name__
        super().__init__(ode_method)

    def __term__(self, t, XP):
        '''
        Define the term for ODE integration.

        Parameters:
        - t: float - The current time.
        - XP: torch.Tensor - The current state vector (position and momentum).

        Returns:
        - torch.Tensor: The term for ODE integration.
        '''

        dX = self.spc.ginv(XP[:4]) @ XP[4:]
        dP = -self.particle.dHmlt(XP[:4], XP[4:], self.eps)

        return torch.cat((dX, dP))


    def evnt(self, t, XP):
        '''
        Event function for ODE integration.

        Parameters:
        - t: float - The current time.
        - XP: torch.Tensor - The current state vector (position and momentum).

        Returns:
        - bool: Whether the event condition is met.
        '''
        return False


    def reg(self, t, XP):
        '''
        Regularization function for ODE integration.

        Parameters:
        - t: float - The current time.
        - XP: torch.Tensor - The current state vector (position and momentum).

        Returns:
        - torch.Tensor: The regularized state vector.
        '''
        return XP