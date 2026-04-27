from abc import abstractmethod
import time

import torch

from bhtrace.geometry import Spacetime, Particle
from bhtrace.utils.odeint import ODEint, ODE
from bhtrace.trajectory.trajectory import Trajectory


class Tracer():
    '''
    Base class for tracing particle trajectories in a given spacetime.

    For 
    '''

    __use_event_fn__ = True

    __g00_tol__ = (-1e3, -1e-1)
    '''Lower and upper bounds for g00'''
    __detg_tol__ = (1e-3, 1e3)
    '''Lower and upper bounds for det(g)'''

    __hmlt_tol__ = 1e-2
    '''Upper bound for hamlitonian violation'''

    __const_dx__ = False

    __use_cached_for_conditions__ = True
    '''If true, cached values will be used for evaluation of event conditions'''

    __tqdm_bar__ = True
    '''If true, tqdm will show up progress bar'''

    def __init__(self,
                 ode_method: str | ODEint = 'Euler',
                 eps: float = 1e-3
                 ):
        '''
        Initialize the Tracer with a specified ODE integration method.
        Parameters:
        - ode_method: str - The ODE integration method to use.
        '''
        self.ode_method = ode_method

        self.default_eps = eps

        self.conditions_XP = {
            'g00': self.__g00_check__,
            'nan': self.__nan_check__,
        }

    def state(self) -> dict:
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

    def forward(self,
                particle: Particle,
                X0: torch.Tensor,
                P0: torch.Tensor,
                T: float,
                nsteps: int = 128,
                r_max: float = 30.0,
                max_proper_t: float = 500.0,
                device: str = 'cpu',
                dtype: torch.dtype = torch.float32,
                eps: float = None
                ) -> Trajectory:
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
        - eps: float - Parameter for numerical differentiation. Default - None \
        (uses self.default_eps value)
        Returns:
        - Trajectory: A Trajectory object containing the traced trajectories.
        '''
        self.particle = particle
        self.spc = particle.spacetime
        self.Ni = X0.shape[0]

        self.eps = eps or self.default_eps

        self.particle._eps = self.eps
        self.particle.spacetime._eps = self.eps

        # Move initial conditions to the target device
        X0 = X0.to(device)
        P0 = P0.to(device)

        # Setup stopping conditions
        self.r_max = torch.tensor([r_max], device=device)
        self.max_proper_t = torch.tensor([max_proper_t], device=device)

        # --- Vectorized Integration ---
        dt = T / nsteps
        self.odeint = ODE(name=self.ode_method, dt=dt, event_fn=self.evnt)
        
        print(f"Starting vectorized integration of {self.Ni} particles with {self.odeint.__class__.__name__}...")
        start_time = time.time()

        solution = self.odeint.forward(
            term=self.__term__,
            Y0=(X0, P0),
            t0=0.0,
            n_steps=nsteps,
            tqdm_bar=self.__tqdm_bar__
        )

        elapsed_time = time.time() - start_time
        print(f"Integration finished in {elapsed_time:.2f} seconds.")

        # Process results
        X = solution['Y'][0]
        P = solution['Y'][1]
        t = solution['t']

        return Trajectory(X, P, t, particle, self, _genuine_steps=solution['active'])
    
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
        - torch.Tensor [batch_size]: A boolean tensor, False where integration should stop.
        '''
        
        for cr in self.conditions_XP.values():
            self.odeint.batch_mask.logical_and_(cr(t, *Y, *dY))
        
    
    def __detg_condition__(self, t, X, P, dX, dP):

        if self.__use_cached_for_conditions__:
            with self.spc.cacher.cache():
                g = self.spc.g(X)
        else:
            g = self.spc.g(X)
        
        detg = torch.linalg.det(g)

        criterion = torch.greater(abs(detg), self.__detg_tol__[0])
        criterion.logical_and_(torch.less(abs(detg), self.__detg_tol__[1]))

        return criterion

    def __hmlt_condition__(self, t, X, P, dX, dP):

        if self.__use_cached_for_conditions__:
            with self.particle.cacher.usecache():
                H = self.particle.hmlt(X, P)
        else:
            H = self.particle.hmlt(X, P)

        dH = H - self.particle.mu
        criterion = torch.less(abs(dH), self.__hmlt_tol__)    
        return criterion
    
    def __nan_check__(self, t, X, Y, dX, dP):

        criterion = ~torch.isnan(dX).all(dim=-1)
        criterion.logical_and_(~torch.isnan(dP).all(dim=-1))
      
        return criterion
    
    def __g00_check__(self, t, X, P, dX, dP):

        if self.__use_cached_for_conditions__:
            with self.spc.cacher.cache():
                g = self.spc.g(X)
        else:
            g = self.spc.g(X)
        
        criterion = torch.greater(g[..., 0, 0], self.__g00_tol__[0])
        criterion.logical_and_(torch.less(g[..., 0, 0], self.__g00_tol__[1]))

        return criterion
    
    def __jump_check__(self, t, X, P, dX, dP):

        criterion = torch.less(abs(dP), self.__jump_tol__).all(dim=-1)
        criterion.logical_and_(torch.less(abs(dX), self.__jump_tol__).all(dim=-1))

        return criterion
    
    def __transform__(self, X, P, dX, dP):
        
        return dX, dP
    
    def to(self, dev = None, dtype = None):

        pass

    def jit(self):
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
        dP = -self.particle.dx_hmlt(X, P)

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
        return torch.ones(X.shape[:-1], dtype=torch.bool)



