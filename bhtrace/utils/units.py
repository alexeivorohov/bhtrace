r"""Submodule for simplifying work with dimensional quantities and unit systems

This module provides tools for defining, manipulating, and converting physical quantities
with associated units. It supports basic mathematical operations and conversions
between different unit systems, adhering to BIPM 2022 choices for fundamental units
and utilizing CODATA values for physical constants.

Attributes
----------
UNIT_EQ_TOLERANCE : float
    Numerical tolerance used for comparing two Quantities or Unit systems for equality.
QUANTITY_REGISTRY : bhtrace.utils.registry.InstanceRegistry
    Registry for mapping known Quantity instances to string keys (aliases).
UNIT_SYSTEM_REGISTRY : bhtrace.utils.registry.InstanceRegistry
    Registry for mapping known UnitSystem instances to string keys (aliases).

Examples
--------
Defining a quantity and translating a quantity:

>>> g = Quantity(9.81, {'L': 1, 'T': -2}, 'si')
>>> g.to('planck')
Quantity(1.581e-35 [L^1, T^-2])

Notes
-----
This implementation follows BIPM 2022 choice of fundamental units [[1]_, [2]_] and takes
most of the constants from `scipy.constants`, which is using the latest CODATA values.

For planck units, charge scale is selected so that :math:`\epsilon_0=1`

References
----------
.. [1] https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-EN.pdf
.. [2] https://en.wikipedia.org/wiki/International_System_of_Units
"""

from typing import Dict, Union, Any, List
import numbers
import math
from collections import Counter
from functools import cached_property

import scipy.constants as constants
import numpy as np

from bhtrace.utils.registry import InstanceRegistry

_PRINCIPAL_UNITS = ["T", "M", "L", "K", "I", "N", "J", "i"]
UNIT_EQ_TOLERANCE = 1e-8


class Quantity:
    """Represents a physical quantity with a value and dimensions.

    Supports basic mathematical operations with floats and other Quantities,
    converison to float and translation to another systems of units.

    Attributes
    ----------
    system : str
        Name of unit system to which this quantity currently binded.
    f : float
        Numerical value of this quantity in current system.
    si : float
        Numerical value of this quantity in SI

    Methods
    -------
    to : (system) -> Quantity
        Moves the quantity to a target unit system.
    """

    def __init__(
        self,
        value: float,
        principal: Dict[str, float],
        system: Union[str, "UnitSystem"] = "si",
    ):
        """Initializes a Quantity object.

        Parameters
        ----------
        value : float
            The numerical value of the quantity in the specified `system`.
        principal : Dict[str, float]
            A dictionary representing the dimensionality of this quantity, where keys
            are principal unit symbols (e.g., 'L', 'T', 'M') and values are their
            corresponding exponents.
        system : str, default='si'
            The name of the unit system in which the `value` is defined.
        """
        self._value = value  # Value is always in SI units
        self._principal = {k: v for k, v in principal.items() if v != 0.0}
        if isinstance(system, str):
            self._system = system
        else:
            self._system = UNIT_SYSTEMS_REGISTRY.typesafe(system)._name

    @property
    def system(self) -> str:
        """Name of the unit system to which this quantity is currently bound.

        Returns
        -------
        str
            The name of the current unit system.
        """
        return self._system

    @cached_property
    def f(self) -> float:
        """Numerical value of this quantity in its current unit system.

        Returns
        -------
        float
            The numerical value of the quantity in the current system.
        """
        return _value_in_system(self, UNIT_SYSTEMS_REGISTRY[self.system])

    @property
    def si(self) -> float:
        """Numerical value of this quantity in SI units.

        Returns
        -------
        float
            The numerical value of the quantity in SI units.
        """
        return self._value

    def to(self, system: Union[str, "UnitSystem"]) -> "Quantity":
        """Translates the quantity to a target unit system.

        Parameters
        ----------
        system : str or UnitSystem
            The target unit system. Can be a string identifier (e.g., 'si', 'planck')
            or an instance of a `UnitSystem`.

        Returns
        -------
        Quantity
            A new `Quantity` object representing the original quantity's value
            expressed in the target unit system, but retaining its original dimensions.
        """
        system = UNIT_SYSTEMS_REGISTRY.typesafe(system)
        return Quantity(self._value, self._principal, system=system._name)

    def __repr__(self) -> str:
        """Returns a string representation of the Quantity.

        Returns
        -------
        str
            A string showing the numerical value and its dimensions,
            e.g., "Quantity(9.81 [L^1, T^-2])".
        """
        sorted_items = sorted(self._principal.items())
        parts = []
        for key, exp in sorted_items:
            if exp == 1.0:
                parts.append(key)
            else:
                parts.append(f"{key}^{exp}")
        unit_str = " ".join(parts) if parts else "dimensionless"
        return f"Quantity({self.system}: {self._value:.4g} [{unit_str}])"

    def _check_dimensionality(self, other: "Quantity") -> bool:
        """Checks if two quantities have compatible unit systems and dimensions.

        Parameters
        ----------
        other : Quantity
            The other Quantity object to compare against.

        Returns
        ------
        bool :
            True if no error raised

        Raises
        ------
        ValueError
            If the quantities are in different unit systems or have different dimensions.
        """
        if self._system != other._system:
            raise ValueError(
                f"Cannot perform operation on quantities in different systems: '{self._system}' and '{other._system}'"
            )
        if self._principal != other._principal:
            raise ValueError(
                f"Cannot perform operation on quantities with different dimensions: {self} and {other}"
            )
        return True

    def __eq__(self, other: 'Quantity') -> bool:
        if abs(self._value - other._value) > UNIT_EQ_TOLERANCE:
            return False
        return self._principal == other._principal

    def __gt__(self, other: 'Quantity') -> bool:
        if self._value > other._value:
            return self._check_dimensionality(other)
        return False

    def __lt__(self, other: 'Quantity') -> bool:
        if self._value < other._value:
            return self._check_dimensionality(other)
        return False

    def __le__(self, other: 'Quantity') -> bool:
        if self._value <= other._value:
            return self._check_dimensionality(other)
        return False

    def __ge__(self, other: 'Quantity') -> bool:
        if self._value >= other._value:
            return self._check_dimensionality(other)
        return False

    def __add__(self, other: "Quantity") -> "Quantity":
        """Adds another Quantity to this quantity.

        Quantities must have the same dimensions and be in the same unit system.

        Parameters
        ----------
        other : Quantity
            The Quantity to add.

        Returns
        -------
        Quantity
            A new Quantity representing the sum.

        Raises
        ------
        ValueError
            If the quantities are in different unit systems or have different dimensions.
        """
        self._check_dimensionality(other)
        return Quantity(self._value + other._value, self._principal, self._system)

    def __sub__(self, other: "Quantity") -> "Quantity":
        self._check_dimensionality(other)
        return Quantity(self._value - other._value, self._principal, self._system)

    def __mul__(self, other: Any) -> "Quantity":
        """Multiplies this quantity by another Quantity or a scalar.

        Parameters
        ----------
        other : Quantity or numbers.Number
            The Quantity or numerical scalar to multiply by.

        Returns
        -------
        Quantity
            A new Quantity representing the product, with combined dimensions.

        Notes
        -----
        If multiplying by a scalar, the dimensions of the original quantity are preserved.
        """
        if isinstance(other, Quantity):
            new_principal = Counter(self._principal)
            new_principal.update(Counter(other._principal))
            return Quantity(
                self._value * other._value, dict(new_principal), self._system
            )
        elif isinstance(other, numbers.Number):
            return Quantity(self._value * other, self._principal, self._system)
        return NotImplemented

    def __rmul__(self, other: Any) -> "Quantity":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Quantity":
        """Divides this quantity by another Quantity or a scalar.

        Parameters
        ----------
        other : Quantity or numbers.Number
            The Quantity or numerical scalar to divide by.

        Returns
        -------
        Quantity
            A new Quantity representing the quotient, with subtracted dimensions.

        Notes
        -----
        If dividing by a scalar, the dimensions of the original quantity are preserved.
        """
        if isinstance(other, Quantity):
            new_principal = Counter(self._principal)
            new_principal.subtract(Counter(other._principal))
            return Quantity(
                self._value / other._value, dict(new_principal), self._system
            )
        elif isinstance(other, numbers.Number):
            return Quantity(self._value / other, self._principal, self._system)
        return NotImplemented

    def __rtruediv__(self, other: Any) -> "Quantity":
        """Implements reverse true division (e.g., scalar / Quantity).

        This method allows a scalar to be on the left-hand side of the division
        operator.

        Parameters
        ----------
        other : Any
            The scalar or Quantity on the left side of the division.

        Returns
        -------
        Quantity
            A new Quantity representing the quotient.
        """
        if isinstance(other, numbers.Number):
            new_principal = {k: -v for k, v in self._principal.items()}
            return Quantity(other / self._value, new_principal, self._system)
        return NotImplemented

    def __pow__(self, power: float) -> "Quantity":
        new_principal = {k: v * power for k, v in self._principal.items()}
        return Quantity(self._value**power, new_principal, self._system)

    def pow(self, power: float) -> "Quantity":
        """Raises the quantity to a given power (alias for `__pow__`).

        Parameters
        ----------
        power : float
            The exponent to raise the quantity to.

        Returns
        -------
        Quantity
            A new Quantity representing the result of the exponentiation.
        """
        return self.__pow__(power)

    def sqrt(self) -> "Quantity":
        """Calculates the square root of the quantity.

        Returns
        -------
        Quantity
            A new Quantity representing the square root.
        """
        return self.pow(0.5)

    def abs(self) -> "Quantity":
        """Calculates the absolute value of the quantity's numerical value.

        The dimensions of the quantity remain unchanged.

        Returns
        -------
        Quantity
            A new Quantity with the absolute numerical value.
        """
        return Quantity(abs(self._value), self._principal, self._system)


QUANTITY_REGISTRY = InstanceRegistry(Quantity)


# Base Units as Quantities with value 1.0
dimensionless = QUANTITY_REGISTRY.r(Quantity(1.0, {}), "dimensionless", "one")
time = QUANTITY_REGISTRY.r(Quantity(1.0, {"T": 1.0}), "second", "sec", "time")
mass = QUANTITY_REGISTRY.r(Quantity(1.0, {"M": 1.0}), "kilogram", "kg", "mass")
length = QUANTITY_REGISTRY.r(Quantity(1.0, {"L": 1.0}), "meter", "m", "length")
current = QUANTITY_REGISTRY.r(Quantity(1.0, {"I": 1.0}), "ampere", "A", "current")
temperature = QUANTITY_REGISTRY.r(
    Quantity(1.0, {"K": 1.0}), "kelvin", "K", "temperature"
)
substance = QUANTITY_REGISTRY.r(
    Quantity(1.0, {"N": 1.0}), "mole", "amount of substance"
)
luminous_intensity = QUANTITY_REGISTRY.r(
    Quantity(1.0, {"J": 1.0}), "candela", "luminous intensity"
)
angle = QUANTITY_REGISTRY.r(Quantity(1.0, {"i": 1.0}), "radian", "rad", "angle")


# Derived Units (calculated from base units)
area = QUANTITY_REGISTRY.r(length**2, "area")
volume = QUANTITY_REGISTRY.r(length**3, "volume")
density = QUANTITY_REGISTRY.r(mass / volume, "density")
frequency = QUANTITY_REGISTRY.r(1.0 / time, "hertz", "frequency")
velocity = QUANTITY_REGISTRY.r(length / time, "velocity")
acceleration = QUANTITY_REGISTRY.r(velocity / time, "acceleration")
momentum = QUANTITY_REGISTRY.r(mass * velocity, "momentum")
force = QUANTITY_REGISTRY.r(mass * acceleration, "newton", "force")
energy = QUANTITY_REGISTRY.r(force * length, "joule", "energy")
power = QUANTITY_REGISTRY.r(energy / time, "watt", "power")
action = QUANTITY_REGISTRY.r(energy * time, "action")
entropy = QUANTITY_REGISTRY.r(energy / temperature, "entropy")
charge = QUANTITY_REGISTRY.r(current * time, "coulomb", "charge")
voltage = QUANTITY_REGISTRY.r(energy / charge, "volt", "voltage")
electric_field = QUANTITY_REGISTRY.r(voltage / length, "electric_field")
magnetic_field = QUANTITY_REGISTRY.r(mass / (charge * time), "magnetic_field")
capacitance = QUANTITY_REGISTRY.r(charge / voltage, "farad", "capacitance")
degree = QUANTITY_REGISTRY.r(math.pi / 180 * angle, "deg", "degree")
solidangle = QUANTITY_REGISTRY.r(angle**2, "ster", "solid angle")
luminosity = QUANTITY_REGISTRY.r(luminous_intensity / solidangle, "lumen", "luminosity")

# Shortcuts & Aliases
m = length
"""Meter"""
K = temperature
"""Kelvin"""
N = force
"""Newton"""
J = energy
"""Joule"""
W = power
"""Watt"""

sec = time
second = time
kg = mass
kilogram = mass
meter = length
coulomb = charge
kelvin = temperature
Hz = frequency
hertz = frequency
newton = force
joule = energy
watt = power
ampere = current
cd = luminous_intensity
candela = luminous_intensity
mole = substance
rad = angle
deg = degree
ster = solidangle
lumen = luminosity
lm = luminosity

# --- Physical Constants ---
c = QUANTITY_REGISTRY.r(constants.c * velocity, "c", "speed of light")
"""Speed of light in vacuum"""

G = QUANTITY_REGISTRY.r(
    constants.G * (length**3) / (mass * time**2), "G", "gravitational constant"
)
"""Gravitational constant"""

h = QUANTITY_REGISTRY.r(constants.h * action, "h", "planck")
"""Planck constant"""

hbar = QUANTITY_REGISTRY.r(constants.hbar * action, "hbar", "planck reduced")
"""Planck reduced constant"""

e = QUANTITY_REGISTRY.r(constants.e * charge, "e", "q_e", "elementary charge")
"""Elementary charge"""

kB = QUANTITY_REGISTRY.r(constants.k * entropy, "kB", "boltzmann")
"""Boltzmann constant"""

N_A = QUANTITY_REGISTRY.r(constants.Avogadro / mole, "N_A", "avogadro")
"""Avogadro constant"""

K_cd = QUANTITY_REGISTRY.r(683.0 * lm / W, "K_cd")
"""Luminous efficacy of monochromatic radiation at 540 THz"""

m_e = QUANTITY_REGISTRY.r(constants.m_e * mass, "m_e", "electron mass")
"""Electron mass"""

m_p = QUANTITY_REGISTRY.r(constants.m_p * mass, "m_p", "proton mass")
"""Proton mass"""

sigma_T = QUANTITY_REGISTRY.r(
    constants.physical_constants["Thomson cross section"][0] * area,
    "sigma_T",
    "thomson cross-section",
)
"""Thomson scattering cross-section"""

sigma_SB = QUANTITY_REGISTRY.r(
    constants.Stefan_Boltzmann * W * m.pow(-2) * K.pow(-4),
    "sigma_SB",
    "stefan-boltzmann",
)
"""Stefan-Boltzmann constant"""

eV = QUANTITY_REGISTRY.r(constants.electron_volt * energy, "eV", "electron volt")
"""Electron volt"""

gauss = QUANTITY_REGISTRY.r(1e-4 * magnetic_field, "gauss")
"""Alternative unit of magnetic field: 10^4 gauss = 1 T"""

year = QUANTITY_REGISTRY.r(constants.year * time, "year")
"""Year in seconds"""

m_sun = QUANTITY_REGISTRY.r(1.988_416e30 * mass, "m_sun", "solar mass")
"""Solar mass"""

schwinger_E = QUANTITY_REGISTRY.r((m_e**2 * c**3) / (e * hbar), "E_crit", "schwinger E")
"""Schwinger critical electric field"""

schwinger_B = QUANTITY_REGISTRY.r((m_e**2 * c**2) / (e * hbar), "B_crit", "schwinger B")
"""Schwinger critical magnetic field"""

eps0 = QUANTITY_REGISTRY.r(
    constants.epsilon_0 * capacitance / length, "eps0", "vacuum permittivity"
)
"""Vacuum electric permittivity"""

mu0 = QUANTITY_REGISTRY.r(
    constants.mu_0 * N * ampere.pow(-2), "mu0", "vacuum permeability"
)
"""Vacuum magnetic permeability"""

k_e = QUANTITY_REGISTRY.r(1.0 / 4.0 / math.pi / eps0, "k_e", "coulomb constant")
"""Coulomb constant"""
alpha = QUANTITY_REGISTRY.r(
    e.pow(2) / 2 / eps0 / h / c, "alpha", "fine-structure constant"
)
"""Fine-structure constant"""

L_edd_sun = QUANTITY_REGISTRY.r(
    4 * math.pi * G * m_p * c / sigma_T, "L_edd_sun", "solar eddington luminosity"
)
"""Eddington solar lumniosity"""

planck_length = QUANTITY_REGISTRY.r((hbar * G * c.pow(-3)).sqrt(), 'planck_L')
planck_mass = QUANTITY_REGISTRY.r((hbar * c / G).sqrt(), 'planck_M')
planck_time = QUANTITY_REGISTRY.r((hbar * G * c.pow(-5)).sqrt(), 'planck_T')
planck_temperature = QUANTITY_REGISTRY.r((hbar * c.pow(5) * kB.pow(-2) / G).sqrt(), 'planck_K')
planck_charge = QUANTITY_REGISTRY.r(e / (4 * math.pi * alpha).sqrt(), 'planck_Q')

light_year = QUANTITY_REGISTRY.r(year * c, 'light_year', 'ly')
ly = light_year

au = QUANTITY_REGISTRY.r(constants.au * meter, 'au', 'astronomical unit')

dminute = QUANTITY_REGISTRY.r(deg / 60, 'deg`', 'degree minute')
dsec = QUANTITY_REGISTRY.r(dminute / 60, 'deg``', 'degree second')
muarcsec = QUANTITY_REGISTRY.r(dsec * 1e-6, 'muarcs', 'muarcsec')
parsec = QUANTITY_REGISTRY.r(constants.parsec * meter, 'pc', 'parsec')

bb_scale = QUANTITY_REGISTRY.r(2 * h / c**2, 'bb_scale')
"""Leading coefficient in the Planck blackbody radiation law"""

bb_power = QUANTITY_REGISTRY.r(h / kB, 'bb_pow') 
"""Coefficient before nu/T in the exponent in the Planck blackbody radiation law"""

m_sgrA = m_sun * 4.297e6
"""Mass of Saggitarius A"""
D_sgrA = 22_996 * ly
"""Distance to Saggitarius A"""

m_M87 = m_sun * 6.5e9 
"""Mass of M87"""
D_M87 = 53.5e6 * ly
"""Distance to M87"""


# TODO: replace principals dict with frozendict in python 3.15
class UnitSystem:
    """Represents a system of units, defining the base units' scaling factors.

    This class allows for defining custom unit systems by specifying the scaling
    factors for each of the principal SI base units (Time, Mass, Length,
    Temperature, Current, Amount of substance, Luminous intensity, and Angle).
    It facilitates conversion between different unit systems.

    Parameters
    ----------
    *aliases : str
        Name aliases of this system. System name will be set to the first element of this sequence.
        Then it will be registered in UNIT_SYSTEM_REGISTRY with this name and aliases.
        If not passed, name will be auto-generated by `.units._system_format` method.
    T : float, default=1.0
        Scaling factor for the base unit of Time (seconds).
    M : float, default=1.0
        Scaling factor for the base unit of Mass (kilograms).
    L : float, default=1.0
        Scaling factor for the base unit of Length (meters).
    K : float, default=1.0
        Scaling factor for the base unit of Temperature (Kelvin).
    I : float, default=1.0
        Scaling factor for the base unit of Electric Current (Ampere).
    N : float, default=1.0
        Scaling factor for the base unit of Amount of substance (mole).
    J : float, default=1.0
        Scaling factor for the base unit of Luminous intensity (candela).
    i : float, default=1.0
        Scaling factor for the base unit of Plane Angle (radian).

    Attributes
    ----------
    name : str
        Unique name of this system.
    time : float
        The scaling factor for time units in this system.
    length : float
        The scaling factor for length units in this system.
    mass : float
        The scaling factor for mass units in this system.
    temperature : float
        The scaling factor for temperature units in this system.
    current : float
        The scaling factor for current units in this system.
    substance : float
        The scaling factor for amount of substance units in this system.
    luminous_intensity : float
        The scaling factor for luminous intensity units in this system.
    angle : float
        The scaling factor for angle units in this system.

    Examples
    --------
    >>> si_system = UnitSystem()
    >>> planck_system = UnitSystem(L=planck_length.f, T=planck_time.f, M=planck_mass.f)
    """

    def __init__(
        self,
        T: float = 1.0,
        M: float = 1.0,
        L: float = 1.0,
        K: float = 1.0,
        I: float = 1.0,
        N: float = 1.0,
        J: float = 1.0,
        i: float = 1.0,
        aliases: List[str] = None,
    ):
        self._principals = {
            "T": T,
            "M": M,
            "L": L,
            "K": K,
            "I": I,
            "N": N,
            "J": J,
            "i": i,
        }
        self.time = T
        self.length = L
        self.mass = M
        self.temperature = K
        self.current = I
        self.substance = N
        self.luminous_intensity = J
        self.angle = i
        aliases = aliases or []
        if len(aliases) == 0:
            aliases = [_system_format(self)]
        self._name = aliases.pop(0)
        if self._name not in UNIT_SYSTEMS_REGISTRY:
            UNIT_SYSTEMS_REGISTRY.r(self, self._name, *aliases)

    def conversion_factor(self, quantity: Union[str, Dict, Quantity]) -> float:
        """Calculates the conversion factor from a given unit in its source system to this unit system.


        Parameters
        ----------
        unit : str, dict, or Quantity
            The unit for which to calculate the conversion factor. Can be:
            - A string alias for a registered Quantity (e.g., 'meter', 'velocity').
            - A dictionary representing the principal dimensions (e.g., {'L': 1, 'T': -1}).
            - A `Quantity` instance.

        Returns
        -------
        float
            The numerical factor by which a value in `from_system` should be multiplied
            to get its equivalent in `self` unit system for the given `unit`.

        Raises
        ------
        ValueError
            If the conversion factor is undefined (division by zero).
        """

        if isinstance(quantity, dict):
            quantity = Quantity(1.0, quantity)
        else:
            quantity = QUANTITY_REGISTRY.typesafe(quantity)

        log_unit_in_si = sum(
            quantity._principal.get(key, 0.0)
            * math.log(self._principals.get(key, 1.0))
            for key in _PRINCIPAL_UNITS
        )
        return math.exp(-log_unit_in_si)

    def __repr__(self) -> str:
        """Returns a string representation of the UnitSystem.

        Returns
        -------
        str
            A string showing the scaling factors for each principal unit.
        """
        fmt = _system_format(self)
        fmt = f" | {fmt}" if fmt != self._name else ""
        return f"UnitSystem({self._name}{fmt})"

    def __eq__(self, other: Any) -> bool:
        """Compares two UnitSystem objects for effective equality.

        Two unit systems are considered equal if their principal unit scaling factors
        are numerically close within `UNIT_EQ_TOLERANCE`.

        Parameters
        ----------
        other : Any
            Object to perform comparison with. Should be `UnitSystem` instance
            or a valid string name of registered `UnitSystem`.

        Returns
        -------
        bool
            True if the unit systems are effectively equal, False otherwise.

        """
        other = UNIT_SYSTEMS_REGISTRY.typesafe(other)
        return _compare_principal_reprs(self._principals, other._principals)

    @property
    def all(self) -> str:
        """Provides a string representation of all registered quantities converted to this unit system.

        Returns
        -------
        str
            A multi-line string listing each registered Quantity and its value
            expressed in the current unit system.
        """
        return "\n".join(
            f"{k}: {v.to(self).f}" for k, v in QUANTITY_REGISTRY.REGISTRY.items()
        )


class DerivedSystem(UnitSystem):
    """Derives a new unit system by setting the value of given physical quantities to 1.

    This method works by constructing a system of linear equations based on the
    dimensions of the input quantities and solving for the scales of the principal
    units (L, M, T, etc.).
    It uses `numpy.linalg.lstsq` to find a solution, which allows for handling
    under-determined systems by providing a minimum-norm solution.

    Parameters
    ----------
    *quantities : Quantity
        One or more `Quantity` objects whose numerical values are to be set to 1
        in the newly derived unit system.

    Raises
    ------
    UserWarning
        If the system is over-determined and inconsistent, a warning is logged
        indicating that the resulting unit system is a least-squares approximation.

    Examples
    --------
    >>> # Create a unit system where c=1 and G=1
    >>> geom_system = DerivedSystem(c, G)
    """

    def __init__(self, *quantities: Quantity, aliases: List[str] = None):
        if not quantities:
            raise ValueError("")

        quantity_list = [QUANTITY_REGISTRY.typesafe(q) for q in quantities]
        principal_units_order = _PRINCIPAL_UNITS

        A = np.array(
            [
                [q._principal.get(p, 0.0) for p in principal_units_order]
                for q in quantity_list
            ]
        )
        b = np.array([math.log(q._value) for q in quantity_list])

        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        if residuals.size > 0 and np.sum(np.abs(residuals)) > 1e-9:
            import logging

            log = logging.getLogger(__file__)
            log.warning(
                "The system is over-determined and likely inconsistent. "
                "The resulting unit system is a least-squares approximation where "
                "the given quantities are close to, but not exactly, 1."
            )

        new_principals_values = {
            p: math.exp(val) for p, val in zip(principal_units_order, x)
        }

        super().__init__(**new_principals_values, aliases=aliases)


def _compare_principal_reprs(p1: Dict[str, float], p2: Dict[str, float]) -> bool:
    """Compares two dictionaries of principal unit scaling factors for effective equality.

    Equality is determined by comparing each corresponding scaling factor within
    a numerical tolerance (`UNIT_EQ_TOLERANCE`).

    Parameters
    ----------
    p1 : Dict[str, float]
        The first dictionary of principal unit scaling factors.
    p2 : Dict[str, float]
        The second dictionary of principal unit scaling factors.

    Returns
    -------
    bool
        True if the scaling factors are effectively equal for all principal units,
        False otherwise.
    """
    for k in _PRINCIPAL_UNITS:
        if p1[k] != p2[k]:
            v = (p1[k] - p2[k]) / (p1[k] + p2[k])
            if not abs(v) < UNIT_EQ_TOLERANCE:
                return False

    return True


def _system_format(system: UnitSystem) -> str:
    return " | ".join([f"{k}:{v:.6g}" for k, v in system._principals.items()])


def _value_in_system(quantity: "Quantity", system: "UnitSystem") -> float:
    """Helper to get the quantity in a given system.

    This method calculates the multiplicative factor that converts a value
    in the given `system` to its equivalent SI value for the specific
    dimensions of this `Quantity`.

    Parameters
    ----------
    quantity : str or Quantity
        Instance of Quantity or it's registered name
    system :  str or UnitSystem
        Instance of UnitSystem or it's registered name

    Returns
    -------
    float
        The numerical value in SI units that represents this
        quantity's in the specified `system`.
    """

    return quantity._value * system.conversion_factor(quantity)


UNIT_SYSTEMS_REGISTRY = InstanceRegistry(UnitSystem)

SI = UnitSystem(aliases=["si", "SI"])
"""International Unit System"""

planck = UnitSystem(
    aliases=["planck"],
    L=planck_length.f,
    T=planck_time.f,
    M=planck_mass.f,
    K=planck_temperature.f,
    I=planck_charge.f / planck_time.f,
    N=1.0,
    J=1.0,
    i=1.0,
)
"""Planck unit system"""


class NaturalGeometric(UnitSystem):
    """Represents a Natural Geometric unit system, often used in General Relativity.

    In this system, fundamental constants like the speed of light (c) and
    gravitational constant (G) are implicitly set to 1. This particular
    implementation defines units based on a given mass scale `M`.

    Parameters
    ----------
    M : float
        The numerical value of the mass scale for this geometric system.
    unit : str or Quantity, default='m_sun'
        The unit of the provided mass scale `M`. Can be a string alias
        (e.g., 'm_sun') or a `Quantity` instance. It must have dimensions of mass.

    Attributes
    ----------
    M_si : float
        The mass scale `M` converted to SI units (kg).
    """

    def __init__(self, M: float, unit: str | Quantity = "m_sun"):
        unit = QUANTITY_REGISTRY.typesafe(unit)
        unit._check_dimensionality(mass)
        M_si = M * unit.si
        L_val = G.si * M_si / (c.si**2)
        T_val = G.si * M_si / (c.si**3)
        M_val = M_si
        K_val = planck_temperature.si
        I_val = (planck_charge / planck_time).si
        super().__init__(
            L=L_val, T=T_val, M=M_val, K=K_val, I=I_val, N=1.0, J=1.0, i=1.0
        )
        self.M_si = M_si

class GRRTUnitSystem(UnitSystem):
    r"""
    Unit system, convenient for GRRT purposes. It is selected so that distances
    are aligned with geometrized units, used during ray-tracing, while radiatively 
    important parameters have "sane" scales.

    Notes
    -----

    Units of this system are scaled so that:
    .. math::

        c = 1, G = 1, M = 1, R_s = 2, T_{\text{char}} = 1
    
    Where :math:`c` is speed of light, :math:`G` is gravitational constant
    :math:`M` is a mass of the central object (e.g. black hole), :math:`R_s`
    is a Schwarzschild radius, :math:T_{\text{char}} - characteristic temperature.
    
    """

    def __init__(self, mass: float, temperature: float = 6.0e3):
        """
        Initializes the GRRTUnitSystem.

        Parameters
        ----------
        mass : float
            Mass of the object in SI units (kg).
        temperature : float, optional
            Characteristic temperature of the object in SI units (K).
            Defaults to 6000 K.
        """
        L = mass * G * c.pow(-2)
        T = L / c 
        super().__init__(L=L.si, T=T.si, M=mass, K=temperature)


def info():
    """Prints information about defined units & units system.

    Returns
    -------
    None
        This function prints directly to stdout and does not return any value.
    """
    print("--- BHTRACE Unit Systems  ---")
    print(f"Defined quantities - {QUANTITY_REGISTRY.info()}")
    print(f"Defined unit systems - {UNIT_SYSTEMS_REGISTRY.info()}")


if __name__ == "__main__":
    info()

    print("""\n--- Registered quantities in planck units ----""")
    print(planck.all)

    print("""\n--- Example - "Schwinger natural units" ----""")
    sys = DerivedSystem(G, c, hbar, schwinger_E, schwinger_B, kB, deg)
    print(sys)
    print(sys.all)
