'''
Submodule with different ray-tracing procedures, including:
- Ray-tracing by solving hamiltonian equations (ptracer)
- Ray-tracing by solving geodesic equation in terms of connection symbols (ctracer)
- Keplerian disks imaging algorithm (ktracer) - planned
'''

from ._base import Tracer, MockTracer
from .ctracer import CTracer
from .ptracer import PTracer