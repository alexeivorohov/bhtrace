from typing import Callable, Tuple

import numpy as np

from bhtrace.utils.registry import InstanceRegistry

class Scaler:
    """
    Baseclass for performing quantity scaling and 
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name: str = None):
        self.func = func
        self.name = name or func.__name__

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.func(x)

    # may be unnecessary when name is clear?
    def ticks_and_lims(self, x: np.ndarray, ) -> np.ndarray:
        raise NotImplementedError

SCALERS_REGISTRY = InstanceRegistry(Scaler)

class LinearScaler(Scaler):

    def __init__(self, scale: float = 1.0):

        name = 'linear'

        if scale == 1.0:
            func = lambda x: x
        else:
            func = lambda x: x*scale
            name += f'[{scale: .3e}]'        

        super().__init__(func, name)


class LogScaler(Scaler):

    def __init__(self, base: float = 10.0, abs: bool = False):
        self.base = base
        self.abs = abs
        self.factor = np.log2(base)
        name = f'log[{base: .3e}]'

        if abs:
            func = lambda x: np.log2(abs(x)) / self.factor
            name += 'abs'
        else:
            func = lambda x: np.log2(x) / self.factor

        super().__init__(func, name=name)

class SymLogScaler(Scaler):

    def __init__(self, base: float = 10.0):
        self.base = base
        self.factor = np.log2(base)
        name = f'symlog[{base: .3e}]'

        super().__init__(self.func, name=name)

    def func(self, x: np.ndarray) -> np.ndarray:
        return np.log2(abs(x)+1.0)*np.sign(x) / self.factor

SCALERS_REGISTRY.register('linear')(LinearScaler())
SCALERS_REGISTRY.register('log10')(LogScaler(10.0))
SCALERS_REGISTRY.register('log2')(LogScaler(2.0))
SCALERS_REGISTRY.register('log')(LogScaler(np.e))
SCALERS_REGISTRY.register('symlog100')(SymLogScaler(100.0))
SCALERS_REGISTRY.register('symlog10')(SymLogScaler(10.0))
SCALERS_REGISTRY.register('symlog2')(SymLogScaler(2.0))
SCALERS_REGISTRY.register('symlog')(SymLogScaler(np.e))


def scale(x: np.ndarray, scale: str = 'linear') -> np.ndarray:

    return SCALERS_REGISTRY[scale](x)


if __name__ == '__main__':

    x = np.linspace(-100.0, 100.0, 25)

    print(scale(x, 'symlog100'))
