import torch
from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod
import inspect
from tqdm import trange

class ODEint(ABC):
    '''
    Base class for ODE solvers.

    This abstract base class defines the interface for ordinary differential equation (ODE) solvers.
    An instance of a solver can be configured with parameters like step size and event functions,
    and can be reused for different ODE integration tasks.

    To create a new solver, you must subclass `ODEint` and implement the `__step__` method,
    and optionally the `__LTE__` method for local truncation error estimation.

    Usage:
    1. Instantiate a solver: `solver = Euler(dt=0.1)`
    2. (Optional) Configure event tracking or adaptive stepping.
    3. Define the ODE term function: `def my_term(t, y): ...`
    4. Solve with `solver.forward()`

    For more control over task preparation and solving, use `solver.attach_task()` and `solver.solve`.
    
    Since this class is designed for solving ODE's for many i.c. in parallel, each tensor in Y0 must have shape at least of (1, 1)

    Event Function Example:
    An `event_fn` can be used to stop integration when a condition is met.
    It should accept `t`, `Y` and `dY` and change components of tracer`s boolean tensor to indicate event occurence for item in the batch.    
    
    def event_fn(t, Y):
    
        self.batch_mask.logical_and_(Y[0] > 0.0)

    

    This example stops integration batch-wise when the first component of Y crosses 0.0

    '''

    __supports_variable_step__ = False
    __variable_step__ = False

    __supports_event_tracking__ = False
    __event_tracking__ = False
  
    __supports_LTE__ = False
    __evaluate_LTE__ = False
    __state_signature__ = ['t', 'Y', 'dY']
    __has_adjoints__ = False


    def __init__(self,
                 dt: float = 1e-3,
                 step_fn: callable = None,
                 event_fn: callable = None,
                 eps: float = 1e-3,
                 ):
        '''
        Initializes the ODE solver with task-independent properties.

        Args:
            dt (float): The time step for integration.
            step_fn (callable, optional): A function for adaptive step size control. Defaults to None.
            event_fn (callable, optional): A function to detect events and stop integration. Defaults to None.
            eps (float, optional): The tolerance for adaptive step size control. Defaults to 1e-3.
        '''
        self.dt = dt
        self.eps = eps
        self.solution: Dict[str, torch.Tensor | Tuple[torch.Tensor] | List] = {}
        
        self.track_LTE()   
        self.set_event(event_fn=event_fn)
        self.set_stepping_strategy(step_fn=step_fn)
         
        self.term = None

        self.n_vars = 0
        self.step_n = 0
        self.t = 0.0
        self.Y0: Tuple[torch.Tensor, ...] | None = None


    def track_LTE(self, on: bool = False):
        if on and not(self.__supports_LTE__):
            print('LTE is not supported for this solver, skipping...')
        else:
            self.__evaluate_LTE__ = on


    def set_event(self, event_fn: callable):
        if self.__supports_event_tracking__ and event_fn is not None:
            self.event_fn = event_fn
            self.event_fn_signature = inspect.signature(event_fn).parameters
            self.__event_tracking__ = True

            if ('LTE' in self.event_fn_signature and not(self.__evaluate_LTE__)):
                self.__event_tracking__ = False
                print('Cannot set up event tracking, since event function accepts LTE as argument and LTE evaluation is not set up; skipping...')
        elif event_fn is not None:
            print('Event tracking is not supported for this solver, skipping...')


    def set_stepping_strategy(self, step_fn: callable):
        '''
        Set step controling method

        Adaptive Step Size Example:
        A `step_fn` can be used for adaptive step size control. It receives the current step size `dt`,
        the local truncation error `LTE`, and the tolerance `eps`.
        
        def step_fn(dt, Y, dY, LTE, eps):
            if LTE is None or LTE <= 0:
                return dt
            safety_factor = 0.9
            return dt * safety_factor * (eps / LTE)**0.5
        '''
        if self.__supports_variable_step__ and step_fn is not None:
            self.step_fn = step_fn
            self.step_fn_signature = inspect.signature(step_fn).parameters
            self.__variable_step__ = True
                    
            if ('LTE' in self.step_fn_signature and not(self.__evaluate_LTE__)):
                self.__variable_step__ = False
                print('Cannot set up adaptive time stepping, since step function accepts LTE as argument and LTE evaluation is not set up; skipping...')
        elif step_fn is not None:
            print('Variable step is not supported for this solver, skipping...')
        

    def add_adjoint_calculation(self, functions: Dict[str, callable]):
        '''
        Add a function or a list of functions over solver's, state, which will be calculated on the go, after each calculation step.
        '''
        self.adjoint_fns = {}
        self.adjoint_fns_signature = {}

        for key, f in functions.items():
            f_signature = inspect.signature(f).parameters
            if not all([k in self.__state_signature__ for k in f_signature]):
                raise ValueError(f"Adjoint function {f} has unsupported signature. Supported arguments are {self.__state_signature__}.")
            else:
                self.adjoint_fns[key] = f
                self.adjoint_fns_signature[key] = f_signature
        
        self.__has_adjoints__ = True

        
    def attach_task(self, term: callable, Y0: Tuple[torch.Tensor, ...], t0: float):
        '''
        :param term: - callable, which represents ODE term and returns tuple(dY)
        :param Y0: - tuple of torch.Tensors, which represenrt initial values of variables. First dimension of each tensor should be a batch dimension of the same size.

        Notice, that step function will unpack values with starred expression, when passing them to the term function.

        This allows more nautral way of operation with variables within a term.

        Example:

        Y0 = (X0, P0)

        def term(t, X, P):
            
            dX = P
            dP = - dH(X, P)

            return dX, dP

        The solver will unpack state tuple Y, then evaluating ODE term. On the first step, it will look like:

        term(t0, *Y0) <-> term(t0, X0, P0)

        This convention is also meant for stepping and event funcitions.

        '''
        self.term = term
        self.n_vars = len(Y0)
        self.batch_size = Y0[0].shape[0]
        self.Y0 = Y0
        self.t0 = t0

    @abstractmethod
    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        pass


    def __post_step__(self, step_n: int, t: float, Y: Tuple[torch.Tensor, ...], dY: Tuple[torch.Tensor, ...]):
        '''
        Process all post-step routines, such as LTE evaluation and event tracking.
        '''
        args = {'t': t, 'Y': Y, 'dY': dY}
        if self.__evaluate_LTE__:
            lte = self.__LTE__(t, Y, dY)
            self.solution['LTE'][step_n, ...] = lte
            args['LTE'] = lte

        if self.__event_tracking__:
            event_args = {k: args[k] for k in self.event_fn_signature if k in args}
            self.event_fn(**event_args)

        if self.__has_adjoints__:
            for f in self.adjoint_fns:
                # TODO: implement this logic
                pass

        if self.__variable_step__:
            args = {'dt': self.dt, 't': t, 'Y': Y, 'dY': dY, 'eps': self.eps}
            if self.__evaluate_LTE__:
                args['LTE'] = self.solution['LTE'][step_n]
            
            step_args = {k: args[k] for k in self.step_fn_signature if k in args}
            self.step_fn(**step_args)


    def solve(self, nsteps: int, track_LTE: bool = False, tqdm_bar=True) -> Dict[str, torch.Tensor | Tuple[torch.Tensor]]:
        '''
        Starts the solving loop for the attached task.
        '''
        
        if self.term is None or self.Y0 is None:
            raise RuntimeError("No task attached. Call attach_task() before solve().")

        # Init solution storage
        self.solution = {}
        self.solution['t'] = torch.zeros(nsteps + 1, device=self.Y0[0].device, dtype=self.Y0[0].dtype)
        self.solution['Y'] = tuple(torch.zeros(self.batch_size, nsteps + 1, *Y0i.shape[1:], device=Y0i.device, dtype=Y0i.dtype) for Y0i in self.Y0)
        self.solution['active'] = torch.zeros(self.batch_size, nsteps + 1, device=self.Y0[0].device, dtype=torch.bool)
        
        # Set initial conditions
        self.solution['t'][0] = self.t0
        for i, y_ in enumerate(self.Y0):
            self.solution['Y'][i][:, 0, ...] = y_

        self.batch_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.Y0[0].device)

        if self.__evaluate_LTE__:
            self.solution['LTE'] = torch.zeros(self.batch_size, nsteps, device=self.Y0[0].device, dtype=self.Y0[0].dtype)

        msg = '\nFinished: Reached maximum number of steps.\n'

        for i in trange(nsteps, disable=tqdm_bar):
            self.solution['active'][:, i+1] = self.batch_mask
            if not torch.any(self.batch_mask):
                # All trajectories stopped, fill rest of solution and break
                last_t = self.solution['t'][i]
                self.solution['t'][i+1:] = last_t
                for j in range(self.n_vars):
                    last_y = self.solution['Y'][j][:, i, ...]
                    # Unsqueeze to allow broadcasting to the remaining steps
                    self.solution['Y'][j][:, i+1:, ...] = last_y.unsqueeze(1)
                msg = '\nFinished: All trajectories have stopped.\n'
                break

            # Get current state for active trajectories
            y = tuple(s[:, i] for s in self.solution['Y'])
            t = self.solution['t'][i]

            y_new, dy = self.__step__(t, y)
            t += self.dt

            # Update solution for active trajectories
            for j in range(self.n_vars):
                # Create a view for the next step
                next_y_slice = self.solution['Y'][j][:, i+1, ...]
                # Copy previous state
                next_y_slice[...] = self.solution['Y'][j][:, i, ...]
                # Update only active ones
                next_y_slice[self.batch_mask, ...] = y_new[j][self.batch_mask, ...]
            
            self.solution['t'][i+1] = t

            # Post-step for new state
            self.__post_step__(step_n=i, t=t, Y=tuple(s[:, i+1] for s in self.solution['Y']), dY=dy)

        if tqdm_bar: print(msg)
        
        return self.solution
    

    def forward(self, 
                term: callable,
                Y0: Tuple[torch.Tensor, ...],
                t0: float, n_steps: int,
                tqdm_bar = True,
                ) -> Dict[str, Tuple[torch.Tensor] | torch.Tensor]:
        '''
        Attach the task and solve it in one call.

        :param term: callable - the ODE term function
        :param Y0: Tuple[torch.Tensor] - initial state
        :param t0: float - initial time
        :param n_steps: int - number of steps to solve
        
        :return: Dict[str, Tuple[torch.Tensor] | torch.Tensor] - the solution dictionary, containing:
        - solution['t'] - torch.Tensor- times
        solution['Y'] - Tuple[torch.Tensor[n_steps, batch_size, ...]] states at each time step
        solution['LTE'] - local truncation error at each time step (if tracked)
        '''
        self.attach_task(term, Y0, t0)
        return self.solve(n_steps, tqdm_bar=tqdm_bar)


    @abstractmethod
    def __LTE__(self, t: float, *Y: torch.Tensor) -> torch.Tensor:
        pass


    def state_dict(self) -> Dict[str, Any]:
        return {
            'dt': self.dt,
            'eps': self.eps,
            'step_n': self.step_n,
            't': self.t,
            'solution': self.solution,
            'n_vars': self.n_vars,
        }


    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.dt = state_dict['dt']
        self.eps = state_dict['eps']
        self.step_n = state_dict['step_n']
        self.t = state_dict['t']
        self.solution = state_dict['solution']
        self.n_vars = state_dict['n_vars']


    def to(self, dev=None, dtype=None):
        '''
        Moves all torch.Tensor quantities of the solver to the specified device and numerical type.
        '''
        if self.Y0 is not None:
            self.Y0 = tuple(y.to(device=dev, dtype=dtype) for y in self.Y0)

        for k, v in self.solution.items():
            if isinstance(v, torch.Tensor):
                self.solution[k] = v.to(device=dev, dtype=dtype)
            elif isinstance(v, list) or isinstance(v, tuple):
                new_v = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        new_v.append(item.to(device=dev, dtype=dtype))
                    elif isinstance(item, tuple):
                        new_v.append(tuple(t.to(device=dev, dtype=dtype) for t in item))
                    else:
                        new_v.append(item)
                
                if isinstance(v, tuple):
                    self.solution[k] = tuple(new_v)
                else:
                    self.solution[k] = list(new_v)
        return self
    
    def behaviour_around(self, t0: float| torch.Tensor, Y0: Tuple[torch.Tensor], nsteps = (2, 2), dt=0.05):
        '''
        Check behaviour around point (t0, Y0) for attached term

        :param t0: time
        :param Y0: Tuple[torch.Tensor] point, for which to check
        :param nsteps: Tuple[int, int] - number of backward and forward (in time) steps from this point.
        :param dt: float - time step
        '''
        
        import uniplot as uplt
        if self.term is None:
            raise RuntimeError("No task attached. Call attach_task() before behaviour_around().")
        self.dt = dt
        self.attach_task(self.term, Y0, t0)

        before = self.solve(nsteps=nsteps[0]+1)
        self.dt = -dt
        after = self.solve(nsteps=nsteps[1]+1)
        
        pass

    def jit(self):
        '''
        Placeholder for jit-compilation of solver
        '''
        return NotImplementedError


class Euler(ODEint):
    __supports_event_tracking__ = True

    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        dY = self.term(t, *Y)
        Y_new = tuple(Y_ + dY[i] * self.dt for i, Y_ in enumerate(Y))
        return Y_new, dY

    def __LTE__(self, t: float, *Y: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

class RK4(ODEint):
    __supports_event_tracking__ = True

    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        dt = self.dt
        k1 = self.term(t, *Y)
        Y_k2 = tuple(Y[i] + k1[i] * dt / 2.0 for i in range(self.n_vars))
        k2 = self.term(t + dt / 2.0, *Y_k2)
        Y_k3 = tuple(Y[i] + k2[i] * dt / 2.0 for i in range(self.n_vars))
        k3 = self.term(t + dt / 2.0, *Y_k3)
        Y_k4 = tuple(Y[i] + k3[i] * dt for i in range(self.n_vars))
        k4 = self.term(t + dt, *Y_k4)
        new_Y = tuple(Y[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) * dt / 6.0 for i in range(self.n_vars))
        return new_Y, k1

    def __LTE__(self, t: float, Y: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
class VCAB4(ODEint):
    '''
    The 4rd order Adams method.
    This implementation uses a fixed step size and RK4 for bootstrapping.
    '''
    __supports_variable_step__ = False
    __supports_event_tracking__ = True

    def __init__(self, dt=0.001, step_fn=None, event_fn=None, eps=0.001):
        super().__init__(dt, step_fn, event_fn, eps)
        self.dY_history = []
        # Use RK4 for bootstrapping
        self.bootstrap_solver = RK4(dt=self.dt)
        # Pass event_fn to bootstrap solver if any
        self.bootstrap_solver.set_event(event_fn)

    def attach_task(self, term: callable, Y0: Tuple[torch.Tensor, ...], t0: float):
        super().attach_task(term, Y0, t0)
        self.bootstrap_solver.attach_task(term, Y0, t0)
        self.dY_history = []

    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if len(self.dY_history) < 2:
            # Bootstrap with RK4 for the first two steps
            self.bootstrap_solver.dt = self.dt  # ensure bootstrap solver has current dt
            Y_new, dY = self.bootstrap_solver.__step__(t, Y)
            return Y_new, dY
        else:
            # Adams-Bashforth 3rd order
            # y_{n+1} = y_n + dt/12 * (23*f_n - 16*f_{n-1} + 5*f_{n-2})
            dY = self.term(t, *Y)  # f_n

            f_nm1 = self.dY_history[-1]
            f_nm2 = self.dY_history[-2]

            Y_new = []
            for i in range(self.n_vars):
                y_new_i = Y[i] + self.dt / 12.0 * (23 * dY[i] - 16 * f_nm1[i] + 5 * f_nm2[i])
                Y_new.append(y_new_i)

            return tuple(Y_new), dY

    def __post_step__(self, step_n: int, t: float, Y: Tuple[torch.Tensor, ...], dY: Tuple[torch.Tensor, ...]):
        self.dY_history.append(dY)
        if len(self.dY_history) > 2:
            self.dY_history.pop(0)

        super().__post_step__(step_n, t, Y, dY)

    def __LTE__(self, t: float, Y: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    def to(self, dev=None, dtype=None):
        super().to(dev, dtype)
        if hasattr(self, 'bootstrap_solver'):
            self.bootstrap_solver.to(dev, dtype)
        # dY_history are tensors, move them too
        if self.dY_history:
            self.dY_history = [tuple(item.to(device=dev, dtype=dtype) for item in dY_tuple) for dY_tuple in self.dY_history]
        return self


class VCABM4(ODEint):
    '''
    The 4th order Adams-Moulton method (predictor-corrector).
    This implementation uses a fixed step size and RK4 for bootstrapping.
    '''
    __supports_variable_step__ = False
    __supports_event_tracking__ = True

    def __init__(self, dt=0.001, step_fn=None, event_fn=None, eps=0.001):
        super().__init__(dt, step_fn, event_fn, eps)
        self.dY_history = []
        # Use RK4 for bootstrapping
        self.bootstrap_solver = RK4(dt=self.dt)
        # Pass event_fn to bootstrap solver if any
        self.bootstrap_solver.set_event(event_fn)

    def attach_task(self, term: callable, Y0: Tuple[torch.Tensor, ...], t0: float):
        super().attach_task(term, Y0, t0)
        self.bootstrap_solver.attach_task(term, Y0, t0)
        self.dY_history = []

    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if len(self.dY_history) < 3:
            # Bootstrap with RK4 for the first three steps
            self.bootstrap_solver.dt = self.dt  # ensure bootstrap solver has current dt
            Y_new, dY = self.bootstrap_solver.__step__(t, Y)
            return Y_new, dY
        else:
            # Predictor (Adams-Bashforth 4th order)
            # y_{n+1}^* = y_n + dt/24 * (55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
            dY = self.term(t, *Y)  # f_n

            f_n = dY
            f_nm1 = self.dY_history[-1]
            f_nm2 = self.dY_history[-2]
            f_nm3 = self.dY_history[-3]

            Y_pred = []
            for i in range(self.n_vars):
                y_pred_i = Y[i] + self.dt / 24.0 * (55 * f_n[i] - 59 * f_nm1[i] + 37 * f_nm2[i] - 9 * f_nm3[i])
                Y_pred.append(y_pred_i)
            Y_pred = tuple(Y_pred)

            # Corrector (Adams-Moulton 4th order)
            # y_{n+1} = y_n + dt/24 * (9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})
            # where f_{n+1} is f(t_{n+1}, y_{n+1}^*)
            dY_pred = self.term(t + self.dt, *Y_pred) # f_{n+1}

            Y_new = []
            for i in range(self.n_vars):
                y_new_i = Y[i] + self.dt / 24.0 * (9 * dY_pred[i] + 19 * f_n[i] - 5 * f_nm1[i] + f_nm2[i])
                Y_new.append(y_new_i)

            return tuple(Y_new), dY

    def __post_step__(self, step_n: int, t: float, Y: Tuple[torch.Tensor, ...], dY: Tuple[torch.Tensor, ...]):
        self.dY_history.append(dY)
        if len(self.dY_history) > 3:
            self.dY_history.pop(0)

        super().__post_step__(step_n, t, Y, dY)

    def __LTE__(self, t: float, Y: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    def to(self, dev=None, dtype=None):
        super().to(dev, dtype)
        if hasattr(self, 'bootstrap_solver'):
            self.bootstrap_solver.to(dev, dtype)
        # dY_history are tensors, move them too
        if self.dY_history:
            self.dY_history = [tuple(item.to(device=dev, dtype=dtype) for item in dY_tuple) for dY_tuple in self.dY_history]
        return self


class SHI(ODEint):
    '''
    Symplectic hamiltionian integrator from paper ()
    '''
    def __init__(self,):
        pass

class Leapfrog(ODEint):
    """
    Leapfrog integrator (a specific symmetric partitioned Runge-Kutta method).
    This is a symplectic integrator, well-suited for Hamiltonian systems.

    It assumes the state Y is a tuple of (positions, momenta), i.e., (X, P),
    and the term function returns (dX/dt, dP/dt).

    The specific method implemented is a 2nd-order symplectic integrator
    for non-separable Hamiltonians of the form H(X, P). It uses a
    kick-drift-kick-like update sequence with three term evaluations.
    """
    __supports_event_tracking__ = True

    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        if self.n_vars != 2:
            raise ValueError("Leapfrog integrator assumes the state Y is a tuple of (positions, momenta).")

        X, P = Y
        dt = self.dt

        # dY at the start of the step. This is also needed for the return value.
        dY_start = self.term(t, X, P)
        _, dP_start = dY_start

        # Kick 1 (half step for momentum P)
        # P_{n+1/2} = P_n + (dt/2) * dP(t_n, X_n, P_n)
        P_half = P + 0.5 * dt * dP_start

        # Drift (full step for position X) using P_half
        # We need dX(t_n + dt/2, X_n, P_{n+1/2}).
        # We re-evaluate the term and take the dX part.
        dX_drift, _ = self.term(t + 0.5 * dt, X, P_half)
        X_new = X + dt * dX_drift

        # Kick 2 (half step for momentum P)
        # We need dP(t_n + dt, X_{n+1}, P_{n+1/2}).
        # We re-evaluate the term at the new position and take the dP part.
        _, dP_end = self.term(t + dt, X_new, P_half)
        P_new = P_half + 0.5 * dt * dP_end

        return (X_new, P_new), dY_start

    def __LTE__(self, t: float, *Y: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

ODE_REGISTRY = {
    'Euler': Euler,
    'RK4': RK4,
    'VCAB4': VCAB4,
    'VCABM4': VCABM4,
    'Leapfrog': Leapfrog
}
    
def ODE(name: str, *args, **kwargs) -> 'ODEint':
        """
        Factory method to create an ODE solver instance by name.
        """
        if name not in ODE_REGISTRY:
            raise ValueError(f"ODE scheme '{name}' is not recognized.\
                             Available schemes are: {list(ODE_REGISTRY.keys())}")

        ode_class = ODE_REGISTRY[name]
        return ode_class(*args, **kwargs)
