import torch
from typing import Tuple, List, Dict, Any
from abc import ABC, abstractmethod
import inspect
from tqdm import trange

class ODEint(ABC):
    '''
    Base class for all ODE solvers. An instance of a solver can be reused for different ODE problems.

    An example of an event_fn that stops integration when the first component of Y crosses 0.0:
    def event_fn(t, Y):
        return Y[0] > 0.0

    An example of a step_fn for adaptive step size control (e.g., for Euler):
    def step_fn(dt, LTE, eps):
        if LTE is None or LTE <= 0:
            return dt
        safety_factor = 0.9
        return dt * safety_factor * (eps / LTE)**0.5
    '''

    __variable_step__ = False
    __event_tracking__ = False
    __supports_variable_step__ = False
    __supports_event_tracking__ = False

    def __init__(self,
                 dt: float,
                 step_fn: callable = None,
                 event_fn: callable = None,
                 eps: float = 1e-3,
                 ):
        '''
        Initializes the ODE solver with task-independent parameters.
        '''
        self.dt = dt
        self.eps = eps
        self._lte_warning_shown = False
        
        self.set_event(event_fn=event_fn)
        self.set_stepping_strategy(step_fn=step_fn)
        
        self.term = None
        self.solution: Dict[str, List] = {}
        self.ntensors = 0
        self.step_n = 0
        self.t = 0.0
        self.Y0: Tuple[torch.Tensor, ...] | None = None

    def set_event(self, event_fn: callable):
        if self.__supports_event_tracking__ and event_fn is not None:
            self.event_fn = event_fn
            self.event_fn_signature = inspect.signature(event_fn).parameters
            self.__event_tracking__ = True

    def set_stepping_strategy(self, step_fn: callable):
        if self.__supports_variable_step__ and step_fn is not None:
            self.step_fn = step_fn
            self.step_fn_signature = inspect.signature(step_fn).parameters
            self.__variable_step__ = True

        else:
            print('Variable step is not supported for this solver, skipping.')

    def attach_task(self, term: callable, Y0: Tuple[torch.Tensor, ...], t0: float):
        self.term = term
        self.ntensors = len(Y0)
        self.batch_dimension = Y0[0].shape[0]
        self.Y0 = Y0
        self.t0 = t0
        self.t = t0
        self.step_n = 0
        self.solution = {'t': [], 'Y': [], 'LTE': []}

    def __post_step__(self, t: float, Y: Tuple[torch.Tensor, ...], dY: Tuple[torch.Tensor, ...], lte: Any) -> bool:
        
        stop = False
        if self.__event_tracking__:
            args = {}
            if 't' in self.event_fn_signature: args['t'] = t
            if 'Y' in self.event_fn_signature: args['Y'] = Y
            if 'dY' in self.event_fn_signature: args['dY'] = dY
            if 'lte' in self.event_fn_signature: args['lte'] = lte
            
            mask = self.event_fn(**args)
            if torch.any(mask):
                # TODO: This print statement is not ideal for batch processing
                # print(f"Event detected at t={t}. Stopping.")
                stop = True

        if self.__variable_step__:
            args = {'dt': self.dt}
            if 't' in self.step_fn_signature: args['t'] = t
            if 'Y' in self.step_fn_signature: args['Y'] = Y
            if 'dY' in self.step_fn_signature: args['dY'] = dY
            if 'LTE' in self.step_fn_signature: args['LTE'] = lte
            args['eps'] = self.eps
            self.dt = self.step_fn(**args)
        
        return stop

    def solve(self, n_steps: int) -> Dict[str, List]:
        if self.term is None or self.Y0 is None:
            raise RuntimeError("No task attached. Call attach_task() before solve().")

        self.solution['t'].append(self.t0)
        self.solution['Y'].append(self.Y0)

        y = self.Y0

        for i in trange(n_steps):
            y_new, dY = self.__step__(self.t, y)
            lte = self.__LTE__(self.t, *y)

            self.t += self.dt
            self.step_n += 1
            
            self.solution['t'].append(self.t)
            self.solution['Y'].append(y_new)
            if lte is not None:
                self.solution['LTE'].append(lte)

            if self.__post_step__(self.t, y_new, dY, lte):
                break
            
            y = y_new
        
        return self.solution

    def forward(self, term: callable, Y0: Tuple[torch.Tensor, ...], t0: float, n_steps: int) -> Dict[str, List]:
        self.attach_task(term, Y0, t0)
        return self.solve(n_steps)

    @abstractmethod
    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        pass

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
            'ntensors': self.ntensors,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.dt = state_dict['dt']
        self.eps = state_dict['eps']
        self.step_n = state_dict['step_n']
        self.t = state_dict['t']
        self.solution = state_dict['solution']
        self.ntensors = state_dict['ntensors']

class Euler(ODEint):
    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        dY = self.term(t, *Y)
        Y_new = tuple(Y[i] + dY[i] * self.dt for i in range(self.ntensors))
        return Y_new, dY

    def __LTE__(self, t: float, *Y: torch.Tensor) -> None:
        if not self._lte_warning_shown:
            print("LTE calculation for Euler is not implemented.")
            self._lte_warning_shown = True
        return None

class RK4(ODEint):
    def __step__(self, t: float, Y: Tuple[torch.Tensor, ...]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        dt = self.dt
        k1 = self.term(t, *Y)
        Y_k2 = tuple(Y[i] + k1[i] * dt / 2.0 for i in range(self.ntensors))
        k2 = self.term(t + dt / 2.0, *Y_k2)
        Y_k3 = tuple(Y[i] + k2[i] * dt / 2.0 for i in range(self.ntensors))
        k3 = self.term(t + dt / 2.0, *Y_k3)
        Y_k4 = tuple(Y[i] + k3[i] * dt for i in range(self.ntensors))
        k4 = self.term(t + dt, *Y_k4)
        new_Y = tuple(Y[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) * dt / 6.0 for i in range(self.ntensors))
        return new_Y, k1

    def __LTE__(self, t: float, *Y: torch.Tensor) -> None:
        if not self._lte_warning_shown:
            print("LTE calculation for RK4 is not implemented.")
            self._lte_warning_shown = True
        return None
    
def ODE(method: str, *args, **kwargs) -> 'ODEint':
        """
        Factory method to create an ODE solver instance by name.
        """
        if method == 'Euler':
            return Euler(*args, **kwargs)
        elif method == 'RK4':
            return RK4(*args, **kwargs)
        else:
            raise NotImplementedError(f"Solver '{method}' is not implemented.")

