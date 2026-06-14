import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt

from bhtrace.scenarios.makers import make_schwarzschild
from bhtrace import Trajectory

# convention: plots are tested in graphics module, not here
# or not?

# --- Fixtures ---

@pytest.fixture(scope="session")
def schwarzschild_2d() -> 'Trajectory':
    return make_schwarzschild('line')

@pytest.fixture(scope="session")
def schwarzschild_3d() -> 'Trajectory':
    return make_schwarzschild('square')


@pytest.fixture
def parameter_dict_example():
    return {

    }

# --- Tests ---

@pytest.mark.skip('Trajectory (de)serialization will be implemented later')
def test_trajectory_from_dict_example():
    pass

@pytest.mark.skip('Trajectory (de)serialization will be implemented later')
def test_trajectory_from_dict_state():
    pass

@pytest.mark.skip('Trajectory (de)serialization will be implemented later')
def test_trajectory_saveload():
    pass


def test_trajectory_interpolation():
    ...

def test_trajectory_to_device_dtype():
    ...

def test_trajectory_coordinate_reprs():
    ...



# --- Plotting tests ---

plot_2d_params = [
    {},
    {},
    {},
]

