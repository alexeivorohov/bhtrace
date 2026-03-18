"""
Defines and initializes project-wide global constants and objects.

Key Globals:
- `CFG_DIR`: Path to the main configuration directory.
- `EPS`: Numeric precision t

"""
import os
import math
import pathlib
import logging
from typing import Any, NamedTuple

import sympy as sp
import sympy.physics.units as units


# --- Logging ---

log = logging.getLogger(__name__)

# --- Constants ---

planck_time = sp.sqrt(units.hbar*units.G/units.c**5)
planck_length = sp.sqrt(units.hbar*units.G/units.c**3)
planck_mass = sp.sqrt(units.hbar*units.c/units.G)
planck_temperature = sp.sqrt(units.hbar*units.c**5/(units.G*units.boltzmann**2))

planck_subs = {
    units.G: 1.0, units.c: 1.0, units.hbar: 1.0, units.boltzmann: 1.0,
}

si_subs = {
    units.G: float(units.G), units.c: float(units.c), 
    units.hbar: float(units.hbar), units.boltzmann: float(units.boltzmann),
}

planck_system = [units.G, units.c, units.hbar, units.boltzmann]

def to_planck_units(expr: Any) -> float:
    return units.convert_to(expr, planck_system).subs(planck_subs)

EPS = os.environ.get('BHTRACE_NUMERIC_EPS', 1e-5)

# --- Paths ---

def _env_var_path_override(name: str, is_dir: bool = False) -> str | None:

    path = os.environ.get(name)
    if path is None:
        return None

    path = pathlib.Path(path)
    assert path.exists(),\
        f"Detected path override by environment variable {name}, but path does not exist {path}"
    
    if is_dir:
        assert path.is_dir(),\
            f"Path override by environment variable {name} is not a directory: {path}"
    else:
        assert path.is_file(),\
            f"Path override by environment variable {name} is not a file: {path}"

    log.info(
        f"Successfully read path override from environment variable {name}, "
        f"set {name} to {path}"
    )

    return path

# --- Path Definitions ---

# The primary configuration directory. Can be overridden by the BHTRACE_CFG_DIR env var.
CFG_DIR =  _env_var_path_override("BHTRACE_CFG_DIR", True) or pathlib.Path(__file__).parent / 'config'
log.info(f'Using configuration directory {CFG_DIR.absolute()}')