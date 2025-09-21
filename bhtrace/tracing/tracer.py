import torch
import time
from abc import abstractmethod
from bhtrace.geometry import Spacetime, Particle
from bhtrace.functional.odeint import ODEint, Euler, RK4
from bhtrace.trajectory.trajectory import Trajectory


class Tracer():
    '''
    Base class for tracing particle trajectories in a given spacetime.

    For 
    '''

    def __init__(self, ode_method='Euler'):
        '''
        Initialize the Tracer with a specified ODE integration method.
        Parameters:
        - ode_method: str - The ODE integration method to use ('Euler', 'RK4').
        '''
        self.ode_method = ode_method
        if ode_method == 'Euler':
            self.ode_solver_class = Euler
        elif ode_method == 'RK4':
            self.ode_solver_class = RK4
        else:
            raise NotImplementedError(f"ODE method '{ode_method}' not supported.")

    def state(self):
        return {
            'tracer': self.__class__.__name__,
            'tracer_param': None,
            'ode_method': self.ode_method,
            'ode_param': None, # self.ode_solver.state()
        }

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
             Y: tuple[torch.Tensor, torch.Tensor],
             dY: tuple[torch.Tensor, torch.Tensor] | None = None,
             ) -> torch.Tensor:
        '''
        Event function to stop integration. Operates on a batch of particles.
        Integration stops for a particle if the function returns True for it.

        Parameters:
        - t: float - The current time.
        - Y: tuple[torch.Tensor, torch.Tensor] - Current state (X, P).
        - dY: tuple[torch.Tensor, torch.Tensor] | None - Current derivatives (dX, dP).
        Returns:
        - torch.Tensor [batch_size]: A boolean tensor, True where integration should stop.
        '''
        X, P = Y
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
        - X0: torch.Tensor [batch_size, 4] - Initial positions.
        - P0: torch.Tensor [batch_size, 4] - Initial momenta.
        - T: float - Simulation time on observer's clock.
        - nsteps: int - Number of steps per trajectory.
        - r_max: float - Distance from center to stop computation.
        - max_proper_t: float - Particle proper time to stop computation.
        - dev: str - Device to perform computation on.
        - eps: float - Parameter for numerical differentiation.
        Returns:
        - Trajectory: A Trajectory object containing the traced trajectories.
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
        X = solution['Y'][0]
        P = solution['Y'][1]

        return Trajectory(X, P, particle, self)
    
    def to(self, dev = None, dtype = None):

        pass


    def jit(self,):
        '''
        Placeholder for jit compilation of tracer
        '''
        return NotImplementedError


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

        dX = torch.einsum('...uv, ...v -> ...u', self.spc.ginv(X), P)
        dP = -self.particle.dHmlt(X, P, self.eps)

        return (dX, dP)


    def evnt(
        self,
        t: float,
        Y: tuple[torch.Tensor, torch.Tensor],
        dY: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        '''
        Event function for ODE integration.

        Parameters:
        - t: float - The current time
        - Y: tuple[torch.Tensor, torch.Tensor] - Current state (X, P).
        - dY: tuple[torch.Tensor, torch.Tensor] | None - Current derivatives (dX, dP).

        Returns:
        - bool: Whether the event condition is met.
        '''
        X, P = Y
        return torch.zeros(X.shape[:-1], dtype=torch.bool)



