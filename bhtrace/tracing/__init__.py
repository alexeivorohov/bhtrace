'''
Submodule with different ray-tracing procedures, including:
- Ray-tracing by solving hamiltonian equations (ptracer)
- Ray-tracing by solving geodesic equation in terms of connection symbols (ctracer)
- Keplerian disks imaging algorithm (ktracer) - planned
'''

from .tracer import Tracer, MockTracer
from .ctracer import CTracer
from .ptracer import PTracer


# Status:
# [X] Tracer baseclass
# [] Baseclass unittest
# [] Per-particle tracing solution
# [] Unittest
# [] Parallel-particle tracing solution
# [] Unittest
# [] Compilable parallel-particle tracing solution
# [] Unittest
# [] Descriptions