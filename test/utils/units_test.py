import pytest
import math
from bhtrace.utils.units import (
    Quantity,
    UnitSystem,
    DerivedSystem,
    NaturalGeometric,
    SI,
    planck,
    c, G, hbar,
    length, mass, time,
    m_sun,
    newton,
    joule,
    kilogram,
    meter,
    second,
    planck_length,
    planck_mass,
    planck_time,
    planck_temperature,
    planck_charge,
    UNIT_SYSTEMS_REGISTRY,
    QUANTITY_REGISTRY
)

def test_quantity_initialization():
    q = Quantity(10.0, {'L': 1, 'T': -1}, 'si')
    assert q.f == 10.0
    assert q.si == 10.0
    assert q.system == 'si'
    assert q._principal == {'L': 1, 'T': -1}

def test_quantity_operations():
    v1 = Quantity(10.0, {'L': 1, 'T': -1})
    v2 = Quantity(5.0, {'L': 1, 'T': -1})
    
    # Addition and Subtraction
    v_sum = v1 + v2
    assert v_sum.f == 15.0
    assert v_sum._principal == v1._principal

    v_diff = v1 - v2
    assert v_diff.f == 5.0
    assert v_diff._principal == v1._principal

    # Mismatched dimensions
    with pytest.raises(ValueError):
        v1 + length

    # Multiplication
    area = length * length
    assert area._principal == {'L': 2}
    
    momentum = mass * v1
    assert momentum.f == 10.0
    assert momentum._principal == {'M': 1, 'L': 1, 'T': -1}

    # Division
    density = mass / (length**3)
    assert density._principal == {'M': 1, 'L': -3}

    # Power
    v_sq = v1 ** 2
    assert v_sq.f == 100.0
    assert v_sq._principal == {'L': 2, 'T': -2}
    
    # Scalar operations
    assert (v1 * 2).f == 20.0
    assert (2 * v1).f == 20.0
    assert (v1 / 2).f == 5.0
    assert (100.0 / v1).f == 10.0

def test_quantity_comparison():
    v1 = Quantity(10, {'L': 1})
    v2 = Quantity(5, {'L': 1})
    v3 = Quantity(10, {'L': 1})

    assert v1 > v2
    assert v2 < v1
    assert v1 >= v3
    assert v1 <= v3

    # Dimensionality check, not value
    assert length == meter 
    assert kilogram != meter

def test_unit_system_conversion():
    # Test SI to Planck
    l_si = 2.0 * meter
    l_planck = l_si.to('planck')
    assert l_planck.f == pytest.approx(l_si.si / planck_length.si)

    # Test back to SI
    l_si_back = l_planck.to('si')
    assert l_si_back.f == pytest.approx(l_si.f)
    
def test_natural_geometric_system():
    # Create a geometric unit system for a solar mass black hole
    m_bh_solar = 1.0
    geo_system = NaturalGeometric(M=m_bh_solar)
    
    # In geometric units, c and G should be 1
    c_geom = c.to(geo_system)
    G_geom = G.to(geo_system)
    
    assert c_geom.f == pytest.approx(1.0)
    assert G_geom.f == pytest.approx(1.0)

    # Test Schwarzschild radius of Sgr A*
    sgra_mass_solar = 4.3e6
    sgra_geo_system = NaturalGeometric(M=sgra_mass_solar)
    
    # Calculate Rs in SI
    rs_si_val = 2 * G.si * (sgra_mass_solar * m_sun.si) / (c.si**2)
    rs_si_quantity = rs_si_val * meter
    
    # Convert to Sgr A* geometric units
    rs_geom = rs_si_quantity.to(sgra_geo_system)
    
    # In a geometric system scaled to its own mass, the Schwarzschild radius should be 2M, where M=1
    # The Schwarzschild radius of SgrA* in a system where M is the mass of SgrA* should be 2.0.
    # To test this, we need to convert Rs to a dimensionless quantity in that system.
    # The length unit in this system is G*M/c^2.
    # So Rs / (length_unit) = (2*G*M_sgra/c^2) / (G*M_sgra/c^2) = 2.
    assert rs_geom.f == pytest.approx(2.0)

def test_derived_system():
    # System where c=1, hbar=1, G=1
    natural_system = DerivedSystem(c, G, hbar)
    
    assert c.to(natural_system).f == pytest.approx(1.0)
    assert G.to(natural_system).f == pytest.approx(1.0)
    assert hbar.to(natural_system).f == pytest.approx(1.0)

    # Check Planck units derived this way
    l_p = length.to(natural_system)
    m_p = mass.to(natural_system)
    t_p = time.to(natural_system)
    
    assert (1/l_p).f == pytest.approx(planck_length.si)
    assert (1/m_p).f == pytest.approx(planck_mass.si)
    assert (1/t_p).f == pytest.approx(planck_time.si)

def test_conversion_factor():
    geo_system = NaturalGeometric(M=1.0)
    
    # Velocity conversion from SI to geometric should be 1/c
    k_vel_si_to_geom = geo_system.conversion_factor(length/time)
    assert k_vel_si_to_geom == pytest.approx(1.0 / c.si)
    
def test_known_quantities():
    # Force: 1 N = 1 kg*m/s^2
    force_derived = kilogram * meter / (second * second)
    assert force_derived.f == newton.f
    assert force_derived._principal == newton._principal
    
    # Energy: 1 J = 1 N*m
    energy_derived = newton * meter
    assert energy_derived.f == joule.f
    assert energy_derived._principal == joule._principal
    
def test_registry_access():
    assert QUANTITY_REGISTRY['meter'] is meter
    assert UNIT_SYSTEMS_REGISTRY['si'] is SI
    assert UNIT_SYSTEMS_REGISTRY['planck'] is planck

def test_quantity_repr():
    q = Quantity(9.81, {'L': 1, 'T': -2})
    assert repr(q) == "Quantity(si: 9.81 [L T^-2])"
    dimless = Quantity(1.0, {})
    assert "dimensionless" in repr(dimless)
