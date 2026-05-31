"""
Defines and initializes project-wide global constants, paths, and configuration 
objects for BHTrace.

This module centralizes system settings such as numeric precision, logging output 
paths, and persistent storage directories (e.g., model configurations and output results). 
Configuration variables can be overridden via environment variables to facilitate 
flexible execution in different environments.

Attributes
----------
EPS : float, defaults to 1e-4
    Default numeric precision throughout the project.
    Controlled by environment variable `BHTRACE_NUMERIC_EPS`.

TRAJECTORY_COORD_REPR_CACHE_SIZE : int, defaults to 2
    Sets the uppper limit on how much coordinate representations
    for Trajectory object to store.
    Controlled by environment variable `BHTRACE_TRAJ_CACHE_SIZE`.

LOG_FACTORY_PARAMS : bool, defaults to False
    Controls whether creation parameters for main objects (Particles, Spacetimes ...) 
    should be outputted to log file.
    Controlled by environment variable `BHTRACE_LOG_FACTORY_PARAMS`.

LOG_FILE : pathlib.Path,
    Path where application logs are written. If not set, logs are outputted only to console.
    Controlled by environment variable `BHTRACE_LOG_FILE`    

CFG_DIR : pathlib.Path, defaults to `.config`
    Path to the main configuration directory, containing yaml configurations of the runs.
    By default points to configurations pre-installed with the package.
    Controlled by environment variable `BHTRACE_CFG_DIR`.
    (TBD: will only be used in future by bhtrace-cli interface)

OUTPUT_DIR: pathlib.Path, defaults to `pathlib.Path.cwd()`
    Default directory for saving all output data (results, checkpoints, etc.). 
    Controlled by BHTRACE_OUTPUT_DIR, or defaults to the current working directory.

_DO_SAVE_TEST_OUTPUTS : bool, defaults to False
    Controls whether plots created during test runs should be saved.
    Useful for reducing disk write operations.

"""
import os
import pathlib
import logging
from bhtrace.utils.operation import _env_var_path_override


log = logging.getLogger(__name__)

# --- Flags ---

LOG_FACTORY_PARAMS: bool = bool(os.environ.get('BHTRACE_LOG_FACTORY_PARAMS', False))
_DO_SAVE_TEST_OUTPUTS: bool = bool(os.environ.get('BHTRACE_DO_SAVE_TEST_OUTPUTS', False))

# --- Constants ---

EPS: float = float(os.environ.get('BHTRACE_NUMERIC_EPS', '1e-4'))
TRAJECTORY_COORD_REPR_CACHE_SIZE: int = int(os.environ.get("BHTRACE_TRAJ_CACHE_SIZE", 4))

# --- Paths ---

CFG_DIR: pathlib.Path = _env_var_path_override("BHTRACE_CFG_DIR", is_dir=True, log=log) or pathlib.Path(__file__).parent / 'config'
# log.info(f'Using configuration directory {CFG_DIR.absolute()}')

LOG_FILE: pathlib.Path | None = _env_var_path_override("BHTRACE_LOG_FILE", log=log)
if LOG_FILE:
    log.info(f"Outputting logs to {LOG_FILE}")

OUTPUT_DIR: pathlib.Path = _env_var_path_override("BHTRACE_OUTPUT_DIR", is_dir=True, log=log) or pathlib.Path.cwd()
log.info(f"Default output directory: {OUTPUT_DIR}")