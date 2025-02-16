'''
Submodule with different ray-tracing procedures, including:
- Ray-tracing by solving hamiltonian equations (ptracer)
- Ray-tracing by solving geodesic equation in terms of connection symbols (ctracer)
- Keplerian disks imagig algorihm (ktracer) - planned
'''

from .tracer import Tracer
from .ctracer import CTracer
from .ptracer import PTracer
from .ntracer import NTracer


# Status:
# [] Tracer baseclass
# [] Baseclass unittest
# [] Per-particle tracing solution
# [] Unittest
# [] Parallel-particle tracing solution
# [] Unittest
# [] Compilable arallel-particle tracing solution
# [] Unittest
# [] Descriptions