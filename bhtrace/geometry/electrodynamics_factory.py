'''
Provides a factory function to create electrodynamics model objects.
'''

from .ed_models import (
    Maxwell,
    EulerHeisenberg,
    BornInfeld,
    ModMax,
    Bardeen,
    ParametricPostMaxwell,
)

ELECTRODYNAMICS_REGISTRY = {
    'Maxwell': Maxwell,
    'EulerHeisenberg': EulerHeisenberg,
    'BornInfeld': BornInfeld,
    'ModMax': ModMax,
    'Bardeen': Bardeen,
    'ParametricPostMaxwell': ParametricPostMaxwell,
}

def create_electrodynamics(name: str, **kwargs):
    '''
    Factory function to create an electrodynamics model object by name.

    Parameters:
    - name: str - The name of the Electrodynamics class to instantiate.
    - **kwargs: Additional keyword arguments to pass to the model's constructor.

    Returns:
    - An instance of the specified Electrodynamics subclass.

    Raises:
    - ValueError: If the specified name is not found in the registry.
    '''
    if name not in ELECTRODYNAMICS_REGISTRY:
        raise ValueError(f"Electrodynamics model '{name}' not recognized. Available models are: {list(ELECTRODYNAMICS_REGISTRY.keys())}")

    ed_class = ELECTRODYNAMICS_REGISTRY[name]
    return ed_class(**kwargs)