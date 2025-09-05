import torch
import os
import pickle
import time
from abc import abstractmethod
from bhtrace.geometry import Spacetime, Particle
from bhtrace.functional.odeint import ODEint, Euler, RK4


class Tracer():
    '''
    Base class for tracing particle trajectories in a given spacetime.
    '''

    def __init__(self, ode_method='Euler'):
        '''
        Initialize the Tracer with a specified ODE integration method.
        Parameters:
        - ode_method: str - The ODE integration method to use ('Euler', 'RK4').
        '''
        if ode_method == 'Euler':
            self.ode_solver_class = Euler
        elif ode_method == 'RK4':
            self.ode_solver_class = RK4
        else:
            raise NotImplementedError(f"ODE method '{ode_method}' not supported.")

    @abstractmethod
    def __term__(self,
                 t: float,
                 X: torch.Tensor,
                 P: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Abstract method to define the term for ODE integration.
        This function should be vectorized to handle batches of particles.
        Parameters:
        - t: float - The current time.
        - X: torch.Tensor [batch_size, 4] - Current positions.
        - P: torch.Tensor [batch_size, 4] - Current momenta.
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: A tuple (dX, dP) representing the derivatives.
        '''
        pass

    def evnt(self,
             t: float,
             X: torch.Tensor, 
             P: torch.Tensor,
             ) -> torch.Tensor:
        '''
        Event function to stop integration. Operates on a batch of particles.
        Integration stops for a particle if the function returns True for it.
        Parameters:
        - t: float - The current time.
        - X: torch.Tensor [batch_size, 4] - Current positions.
        - P: torch.Tensor [batch_size, 4] - Current momenta.
        Returns:
        - torch.Tensor [batch_size]: A boolean tensor, True where integration should stop.
        '''
        # Stop if radial coordinate > r_max or proper time > max_proper_t
        cr1 = torch.greater(X[:, 0], self.max_proper_t)
        cr2 = torch.greater(torch.abs(X[:, 1]), self.r_max)
        return cr1 | cr2

    def forward(self,
                particle: Particle,
                X0: torch.Tensor,
                P0: torch.Tensor,
                T: float,
                nsteps: int = 128,
                r_max: float = 30.0,
                max_proper_t: float = 500.0,
                dev: str = 'cpu',
                eps: float = 1e-3):
        '''
        Trace particle trajectories in parallel.
        Parameters:
        - particle: Particle - The particle to trace.
        - X0: torch.Tensor [n, 4] - Initial positions.
        - P0: torch.Tensor [n, 4] - Initial momenta.
        - T: float - Simulation time on observer's clock.
        - nsteps: int - Number of steps per trajectory.
        - r_max: float - Distance from center to stop computation.
        - max_proper_t: float - Particle proper time to stop computation.
        - dev: str - Device to perform computation on.
        - eps: float - Parameter for numerical differentiation.
        Returns:
        - tuple: (X, P) where X and P are tensors of shape (nsteps+1, n, 4).
        '''
        self.particle = particle
        self.spc = particle.spacetime
        self.Ni = X0.shape[0]
        self.eps = eps

        # Move initial conditions to the target device
        X0 = X0.to(dev)
        P0 = P0.to(dev)

        # Setup stopping conditions
        self.r_max = torch.tensor([r_max], device=dev)
        self.max_proper_t = torch.tensor([max_proper_t], device=dev)

        # --- Vectorized Integration ---
        dt = T / nsteps
        odeint = self.ode_solver_class(dt=dt, event_fn=self.evnt)
        
        print(f"Starting vectorized integration of {self.Ni} particles with {self.ode_solver_class.__name__}...")
        start_time = time.time()

        solution = odeint.forward(
            term=self.__term__,
            Y0=(X0, P0),
            t0=0.0,
            n_steps=nsteps
        )

        elapsed_time = time.time() - start_time
        print(f"Integration finished in {elapsed_time:.2f} seconds.")

        # Process results
        # solution['Y'] is a list of tuples [(X0, P0), (X1, P1), ...]
        # We stack them to get tensors of shape (nsteps+1, batch_size, 4)
        x_steps = [item[0] for item in solution['Y']]
        p_steps = [item[1] for item in solution['Y']]

        self.X = torch.stack(x_steps)
        self.P = torch.stack(p_steps)

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

    def __term__(self, t, X, P):
        '''
        Define the term for ODE integration.

        Parameters:
        - t: float - The current time
        - X: torch.Tensor - Current coordinates
        - P: torch.Tensor - Current momentum

        Returns:
        - Tuple[torch.Tensor]: The term for ODE integration.
        '''

        dX = self.spc.ginv(X) @ P
        dP = -self.particle.dHmlt(X, P, self.eps)

        return (dX, dP)


    def evnt(self,
             t,
             X,
             P
             ):
        '''
        Event function for ODE integration.

        Parameters:
        - t: float - The current time
        - X: torch.Tensor - Current coordinates
        - P: torch.Tensor - Current momentum


        Returns:
        - bool: Whether the event condition is met.
        '''
        return False



