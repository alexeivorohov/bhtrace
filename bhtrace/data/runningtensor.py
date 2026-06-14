from typing import Any, Optional, Dict, Literal, Tuple, List, Iterable, NamedTuple
from dataclasses import dataclass, field

import torch

class RunningTensor:
    """Data structure for handling masked tensors changing over time
    
    User is responsible for storing the masks.
    
    """
    trace: bool
    _diffs: List[torch.Tensor]
    x: torch.Tensor
    shape: Tuple[int]

    def __init__(
        self,
        x0: torch.Tensor, 
        trace: bool = False,
        _diffs: Optional[List[torch.Tensor]] = None, 
    ):   
        self.trace = trace

        if _diffs is None:
            _diffs = []
        self._diffs = _diffs
        self._x0 = x0.clone()
        self.x = x0
        self.steps = 0
        self.shape = x0.shape

    def update(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """Performs masked update and stores the update data
        Parameters
        ----------
        x : torch.Tensor
            New value
        mask : torch.Tensor
            Mask for `self.x` to place new value at
        """
        self.x[mask] = x
        if self.trace:
            self.steps += 1
            self._diffs.append(x.clone())

    def clean(self):
        self.steps = 0
        self._diffs = []

    def inflate(self, masks: List[torch.Tensor]) -> torch.Tensor:

        outp = torch.zeros([*self.shape, masks.__len__()+1], dtype=self.x.dtype, device=self.x.device)
        outp[..., 0]  = self._x0
        
        j = 0
        for i in range(1, masks.__len__()):
            if masks[i].any():
                # print(self._diffs[j].shape, masks[i].sum())
                outp[masks[i], ..., i] = self._diffs[j]
                j += 1
                neg = masks[i].logical_not().squeeze()
                outp[neg, ..., i] = outp[neg, ..., i-1]
            else:
                outp[..., i] = outp[..., i-1]

        return outp

    def to(self, device, dtype) -> None:
        ...

    def __len__(self):
        return self.steps + 1
        



