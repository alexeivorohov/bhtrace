'''
Submodule for hanlding several routines:
- Image plotting and graphical pre-sets
- Differentiation
- ODE Integration
- Linear and tensor algebra
- Rootfinfding
- etc
'''

from .routines import *
from .odeint import *

from .diff import *
from .linalg import *
from .transform import *
from .caching import *
from .operation import *
from .log import Logger, LOG