from typing import Dict, Union
import scipy.constants as constants
import math
from collections import Counter

from bhtrace.utils.registry import InstanceRegistry


_PRINCIPAL_UNITS = ['T', 'M', 'L', 't', 'Q']
_SUN_MASS = 1.988_416e30

# --- Units ---

class Unit:
    """Represents a physical unit as a combination of principal units."""

    def __init__(self, principal: Dict[str, float]):
        # Store only non-zero exponents
        self.principal = {k: v for k, v in principal.items() if v != 0.0}

    def to_si(self, system: 'UnitSystem') -> float:
        """Calculates the SI value of one unit in a given system."""
        prods = [
            math.pow(system.principals.get(key, 1.0), self.principal[key])
            for key in self.principal
        ]
        return math.prod(prods)

    def __repr__(self):
        # Sort for a consistent representation
        sorted_items = sorted(self.principal.items())
        parts = []
        for key, exp in sorted_items:
            if exp == 1.0:
                parts.append(key)
            else:
                parts.append(f"{key}^{exp}")
        return f"Unit({', '.join(parts)})"

    def __mul__(self, other: 'Unit') -> 'Unit':
        new_principal = Counter(self.principal)
        new_principal.update(Counter(other.principal))
        return Unit(dict(new_principal))

    def __truediv__(self, other: 'Unit') -> 'Unit':
        new_principal = Counter(self.principal)
        new_principal.subtract(Counter(other.principal))
        return Unit(dict(new_principal))

    def __pow__(self, power: float) -> 'Unit':
        new_principal = {k: v * power for k, v in self.principal.items()}
        return Unit(new_principal)

UNITS_REGISTRY = InstanceRegistry(Unit)

# Base Units
time = Unit({'T': 1.0})
mass = Unit({'M': 1.0})
length = Unit({'L': 1.0})
charge = Unit({'Q': 1.0})
temperature = Unit({'t': 1.0})
dimensionless = Unit({})

UNITS_REGISTRY.register('time')(time)
UNITS_REGISTRY.register('mass')(mass)
UNITS_REGISTRY.register('length')(length)
UNITS_REGISTRY.register('charge')(charge)
UNITS_REGISTRY.register('temperature')(temperature)
UNITS_REGISTRY.register('dimensionless')(dimensionless)

# Derived Units
velocity = length / time
UNITS_REGISTRY.register('velocity')(velocity)
acceleration = velocity / time
UNITS_REGISTRY.register('acceleration')(acceleration)
force = mass * acceleration
UNITS_REGISTRY.register('force')(force)
energy = force * length
UNITS_REGISTRY.register('energy')(energy)
voltage = energy / charge
UNITS_REGISTRY.register('voltage', aliases=['volt'])(voltage)
magnetic_field = mass / (charge * time)
UNITS_REGISTRY.register('magnetic_field')(magnetic_field)
action = energy * time
UNITS_REGISTRY.register('action')(action)
entropy = energy / temperature
UNITS_REGISTRY.register('entropy')(entropy)
area = length ** 2
UNITS_REGISTRY.register('area')(area)
G_unit = (length ** 3) / (mass * time ** 2)
UNITS_REGISTRY.register('G_unit')(G_unit)


# --- Unit Systems ---

class UnitSystem:
    def __init__(self, L: float = 1.0, T: float = 1.0, M: float = 1.0, t: float = 1.0, Q: float = 1.0):
        """Initializes a unit system with the SI values of its base units."""
        self.principals = {'L': L, 'T': T, 'M': M, 't': t, 'Q': Q}

    def conversion_factor(self, unit_def: Union[str, Dict[str, float]], from_system: str = 'SI') -> float:
        """
        Calculates the conversion factor for a given unit from another system
        to this one.

        The factor k is defined as: value_in_this_system = k * value_in_from_system.

        Parameters
        ----------
        unit_def : Union[str, Dict[str, float]]
            The unit to be converted. Can be a name from the UNITS_REGISTRY
            (e.g., 'velocity') or a dictionary of principal dimensions 
            (e.g., {'L': 1, 'T': -1}).
        from_system : str, optional
            The name of the source unit system, by default 'SI'.

        Returns
        -------
        float
            The conversion factor.
        """
        if isinstance(unit_def, dict):
            unit = Unit(unit_def)
        elif isinstance(unit_def, str):
            if unit_def not in UNITS_REGISTRY:
                raise KeyError(f"Unit '{unit_def}' not found in UNITS_REGISTRY.")
            unit = UNITS_REGISTRY[unit_def]
        else:
            raise TypeError("unit_def must be a string (unit name) or a dict (principal dimensions).")

        if from_system not in UNIT_SYSTEMS_REGISTRY:
            raise KeyError(f"Source system '{from_system}' not found in UNIT_SYSTEMS_REGISTRY.")
        
        source_system = UNIT_SYSTEMS_REGISTRY[from_system]
        target_system = self

        val_in_si_from_source = unit.to_si(source_system)
        val_in_si_from_target = unit.to_si(target_system)

        if val_in_si_from_target == 0.0:
            raise ValueError("Conversion factor to the target system is undefined (division by zero).")

        return val_in_si_from_source / val_in_si_from_target

UNIT_SYSTEMS_REGISTRY = InstanceRegistry(UnitSystem)
SI = UnitSystem()
UNIT_SYSTEMS_REGISTRY.register('SI', aliases=['si'])(SI)


# --- Physical Constants ---

class PhysicalConstant:
    def __init__(self, value: float, unit_name: str, system_name: str = 'SI'):
        self.value = value
        self.unit_name = unit_name
        self.system_name = system_name

    @property
    def unit(self) -> Unit:
        return UNITS_REGISTRY[self.unit_name]

    @property
    def system(self) -> UnitSystem:
        return UNIT_SYSTEMS_REGISTRY[self.system_name]

    def to_system(self, target_system_name: str) -> 'PhysicalConstant':
        """Converts the constant to a different unit system."""
        if target_system_name not in UNIT_SYSTEMS_REGISTRY:
             raise KeyError(f"Target system '{target_system_name}' is not registered.")
        target_system = UNIT_SYSTEMS_REGISTRY[target_system_name]
        
        si_value = self.value * self.unit.to_si(self.system)
        
        new_value = si_value / self.unit.to_si(target_system)
        
        return PhysicalConstant(new_value, self.unit_name, target_system_name)
    
    def __repr__(self) -> str:
        return f"{self.value} [{self.unit_name}] in ({self.system_name})"

CONSTANTS_REGISTRY = InstanceRegistry(PhysicalConstant)

c = PhysicalConstant(constants.c, 'velocity')
G = PhysicalConstant(constants.G, 'G_unit')
h = PhysicalConstant(constants.h, 'action')
hbar = PhysicalConstant(constants.h, 'action')
e = PhysicalConstant(constants.e, 'charge')
kB = PhysicalConstant(constants.k, 'entropy')
m_e = PhysicalConstant(constants.m_e, 'mass')
m_p = PhysicalConstant(constants.m_p, 'mass')
sigma_T = PhysicalConstant(constants.physical_constants['Thomson cross section'][0], 'area')
eV = PhysicalConstant(constants.electron_volt, 'energy')
gauss = PhysicalConstant(1e-4, 'magnetic_field')

CONSTANTS_REGISTRY.register('c', aliases=['speed_of_light'])(c)
CONSTANTS_REGISTRY.register('G', aliases=['gravitational_constant'])(G)
CONSTANTS_REGISTRY.register('h', aliases=['planck'])(h)
CONSTANTS_REGISTRY.register('hbar', aliases=['reduced_planck'])(hbar)
CONSTANTS_REGISTRY.register('e', aliases=['elementary_charge'])(e)
CONSTANTS_REGISTRY.register('kB', aliases=['boltzmann'])(kB)
CONSTANTS_REGISTRY.register('m_e', aliases=['electron_mass'])(m_e)
CONSTANTS_REGISTRY.register('m_p', aliases=['proton_mass'])(m_p)
CONSTANTS_REGISTRY.register('sigma_T', aliases=['thomson_cross_section'])(sigma_T)
CONSTANTS_REGISTRY.register('eV', aliases=['electron_volt'])(eV)
CONSTANTS_REGISTRY.register('gauss')(gauss)


class NaturalGeometric(UnitSystem):
    """
    Class for conversion between SI and natural geometric units.
    In this system, G = c = 1. The length scale is set by the gravitational
    radius of a central object of mass M, r_g = GM/c^2.
    We also set kB = 1, and 1/(4*pi*epsilon_0) = 1.
    """

    def __init__(self, M: float, sun: bool = True):
        """
        Parameters
        ----------
        M : float
            Mass of the central object, in SI units [kg]. If `sun` is True,
            M is interpreted as solar masses.
        sun : bool
            If true, multiplies M by the mass of the sun.
        """
        if sun:
            M_si = M * _SUN_MASS
        else:
            M_si = M

        L_val = constants.G * M_si / (constants.c ** 2)
        T_val = constants.G * M_si / (constants.c ** 3)
        M_val = M_si
        t_val = M_val * (constants.c ** 2) / constants.k
        Q_val = M_si * math.sqrt(constants.G * 4 * math.pi * constants.epsilon_0)
        
        super().__init__(L=L_val, T=T_val, M=M_val, t=t_val, Q=Q_val)

        self.M_si = M_si
        self.R_s = 2.0 * L_val
        self._length_scale = 1.0 / L_val
        self._time_scale = 1.0 / T_val
        self._mass_scale = 1.0 / M_val
        self._temp_scale = 1.0 / t_val
        self._charge_scale = 1.0 / Q_val
    

if __name__ == "__main__":
    print("--- BHTRACE Unit System ---")
    print(f"Defined units: {list(UNITS_REGISTRY.keys())}")
    print(f"Defined constants: {list(CONSTANTS_REGISTRY.keys())}")
    
    print("\n--- Unit Operations ---")
    v = length / time
    print(f"length / time = {v}")
    a = v / time
    print(f"velocity / time = {a}")
    e = mass * v**2
    print(f"mass * velocity^2 = {e}")
    
    print("\n--- Physical Constants in SI ---")
    c_si = CONSTANTS_REGISTRY['c']
    G_si = CONSTANTS_REGISTRY['G']
    print(f"c = {c_si}")
    print(f"G = {G_si}")
    
    # Create unit systems for demonstration
    m_bh_solar = 1.0
    geo_name = f"Geom(M={m_bh_solar:.1f}sun)"
    if geo_name not in UNIT_SYSTEMS_REGISTRY:
        geo_system = NaturalGeometric(M=m_bh_solar, sun=True)
        UNIT_SYSTEMS_REGISTRY.register(geo_name)(geo_system)

    sgra_mass_solar = 4.3e6
    sgra_geo_name = f"Geom(SgrA*)"
    if sgra_geo_name not in UNIT_SYSTEMS_REGISTRY:
        sgra_geo_system = NaturalGeometric(M=sgra_mass_solar, sun=True)
        UNIT_SYSTEMS_REGISTRY.register(sgra_geo_name)(sgra_geo_system)

    print("\n--- Unit System Conversion Example ---")
    print(f"Created & registered unit system: '{geo_name}'")
    c_geom = c_si.to_system(geo_name)
    G_geom = G_si.to_system(geo_name)
    print(f"Value of c in '{geo_name}': {c_geom.value:.2f}")
    print(f"Value of G in '{geo_name}': {G_geom.value:.2f}")

    sgra_rs_si_val = 2 * constants.G * (sgra_mass_solar * _SUN_MASS) / (constants.c**2)
    rs_constant_si = PhysicalConstant(sgra_rs_si_val, 'length')
    rs_constant_geom = rs_constant_si.to_system(sgra_geo_name)
    
    print(f"\nSchwarzschild radius of Sgr A* ({sgra_rs_si_val:.3e} m) in its own geometric units:")
    print(f"  Value is: {rs_constant_geom.value:.2f} [{rs_constant_geom.unit_name}] in ({rs_constant_geom.system_name})")
    print("  (Expected value is 2.0, since R_s = 2*r_g and r_g is the length unit for this system)")

    print("\n--- Conversion Factor Example ---")
    geo_system_instance = UNIT_SYSTEMS_REGISTRY[geo_name]
    
    # Get conversion factor for velocity from SI to geometric units
    k_vel_si_to_geom = geo_system_instance.conversion_factor('velocity', from_system='SI')
    print(f"Velocity conversion factor (SI -> {geo_name}): {k_vel_si_to_geom:.4g}")
    print(f"(Should be 1/c = {1/constants.c:.4g})")

    # Get conversion factor for velocity from geometric units to SI
    k_vel_geom_to_si = SI.conversion_factor('velocity', from_system=geo_name)
    print(f"Velocity conversion factor ({geo_name} -> SI): {k_vel_geom_to_si:.4g}")
    print(f"(Should be c = {constants.c:.4g})")
    
    # Test with a dict for area
    k_area_si_to_geom = geo_system_instance.conversion_factor({ 'L': 2.0 }, from_system='SI')
    L_val = geo_system_instance.principals['L']
    print(f"Area conversion factor (SI -> {geo_name}): {k_area_si_to_geom:.4g}")
    print(f"(Should be 1/L_val^2 = {1/(L_val**2):.4g})")


    print("\n --- Main example ---")
    geo_system = NaturalGeometric(1.0)
    k_es_factor = geo_system.conversion_factor({'M': -1, 'L': 2})
    print(f"Conversion factor for electron scattering coefficient: {k_es_factor}")
    print(f"Typical value of electron scattering coefficient in geom.units: {0.5e-3*k_es_factor}")
    

