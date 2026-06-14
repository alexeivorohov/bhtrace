'''
Submodule for hanlding several routines:
- Differentiation
- ODE Integration
- Linear and tensor algebra
- Rootfinfding
- Physical constants and unit systems
- etc
'''
from . import units

from .routines import *
from .odeint import *

from .diff import *
from .linalg import *
from .transform import *
from .caching import * # DEPRECATING
from .operation import *
from .log import Logger, LOG
from .registry import (
    Registry, # OLD NAME
    CallableRegistry,
    InstanceRegistry,
    ClassRegistry,
)