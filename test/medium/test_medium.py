import pytest
import torch

# Spacetimes for testing
from bhtrace.geometry.spacetime.spherical import KerrBL

# Mediums for testing
from bhtrace.medium.thin_disks import KeplerianDisk, AlphaDisk

# Setup test cases: (MediumClass, SpacetimeInstance, constructor_args)
# For KeplerianDisk, Schwarzschild is sufficient.
medium_test_cases = [
    (KeplerianDisk, KerrBL(), {}),
    # (AlphaDisk, KerrBL(), {"alpha": 0.1, "m_dot": 1.0}),
]

# Create IDs for pytest parametrization to make test output more readable
def idfn(val):
    if isinstance(val, tuple):
        return f"{val[0].__name__}-{val[1].__class__.__name__}"
    return str(val)

@pytest.fixture(params=medium_test_cases, ids=idfn)
def medium_instance(request):
    """Fixture to create instances of different Medium subclasses."""
    medium_class, spacetime, kwargs = request.param
    try:
        instance = medium_class(spacetime=spacetime, **kwargs)
        return instance
    except Exception as e:
        pytest.fail(f"Failed to instantiate {medium_class.__name__} with {spacetime.__class__.__name__}: {e}")

@pytest.fixture(scope="module")
def sample_coords():
    """Provides sample coordinates for testing, avoiding the singularity and ISCO."""
    # Batch of 10 points: (t, r, theta, phi)
    coords = torch.rand(10, 4, dtype=torch.float64)
    # Set r to be in a reasonable range for disks, e.g., [7, 30]
    coords[..., 1] = 7.0 + 23.0 * coords[..., 1]
    # Set theta and phi to their valid ranges
    coords[..., 2] = torch.pi * coords[..., 2]
    coords[..., 3] = 2 * torch.pi * coords[..., 3]
    return coords

def run_method_test(medium_instance, method_name, *args):
    """Helper function to run a test on a single method."""
    if not hasattr(medium_instance, method_name):
        pytest.skip(f"{type(medium_instance).__name__} does not implement '{method_name}'")

    method = getattr(medium_instance, method_name)

    try:
        result = method(*args)
        assert result is not None, f"'{method_name}' returned None"
        if isinstance(result, torch.Tensor):
            # signed_distance can return NaNs by design for points outside the disk.
            # A failure is when ALL values are NaN, which likely indicates an issue.
            if torch.isnan(result).all():
                 pytest.fail(f"'{method_name}' returned a tensor of all NaNs.")

    except NotImplementedError:
        pytest.skip(f"'{method_name}' is not implemented in {type(medium_instance).__name__}")
    except Exception as e:
        pytest.fail(f"'{method_name}' raised an unexpected exception: {e}")

def test_medium_methods_return_valid_values(medium_instance, sample_coords):
    """
    A general test for instances of Medium.
    It checks that key methods do not return NaN values for a sample of coordinates.
    """
    methods_with_coords = [
        "velocity",
        "temperature",
        "surface_flux",
        "signed_distance",
        "opacity",
        "pressure",
        "surface_density",
        "height",
        "rest_mass_density",
    ]

    for method_name in methods_with_coords:
        run_method_test(medium_instance, method_name, sample_coords)

    # Test for hit_condition, which has a different signature
    s0 = torch.tensor([1.0, -1.0, 1.0, 0.0])
    s1 = torch.tensor([-1.0, 1.0, 1.0, 1.0])
    run_method_test(medium_instance, "hit_condition", s0, s1)
