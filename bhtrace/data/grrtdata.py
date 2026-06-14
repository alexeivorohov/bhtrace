from typing import Any, Optional, Dict, Literal, Tuple, List, Iterable, NamedTuple
from dataclasses import dataclass, field

import torch

from .runningtensor import RunningTensor

@dataclass
class GRRTData:
    hits: List[torch.Tensor]
    """Each item represents bool tensor mask for i-th step, pointing which points have contributed into radiation"""
    z: RunningTensor
    fluxes: List[RunningTensor]
