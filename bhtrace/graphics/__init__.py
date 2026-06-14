"""
This module contains visualization utilities for `bhtrace`.

It is recommended to import this entire as:
```python
import bhtrace.graphics as bhg

bhg.plot2d(...)
```

Most of the functions are shipped with multiple backends, which can be 
selected by passing the `backend` argument. The default backend is `matplotlib`, 
but other backends are available for some functions.

Methods
-------

plot2d : 
    General function to plot 2d trajectories, supports different projections, coloring by scalar value, horizon plotting and e.t.c.

plot3d :
    General function to plot 2d trajectories, supports coloring by scalar value, horizon plotting and e.t.c.

hist : 
    Function to study the distribution of some scalar quantity (e.g. energy deviation)

ridge : 
    Function to study the distribution cgange of some scalar quantity (e.g. energy deviation) along time / batch.

lensing_curve :
    Preset for plotting declination angle vs. impact parameter curve.


Examples
--------



Submodules
----------

traj2d : 
    Implements backends of `plot2d` function

traj3d :
    Implements backends of `plot3d` function

lensing : 
    Implements backends of `lensing_curve` function

histogram :
    Implements backends of `hist` and `ridge` functions

horizons :
    Responsible for 2d and 3d horizon plotting

coloring : 
    Controls color presets and provides methods to draw colored lines

presets : 
    Provides style presets and constants

utils : 
    Provides utility methods and clases for this module

uniplot_wraps : (visible as bhg.uniplot) 
    Wraps methods of Uniplot package into UniFigure and UniAxes classes to provide
    a stateful, matplotlib-like plots in terminal

"""

# --- Public API ---

from .traj2d import plot2d
from .traj3d import plot3d, scatter3d
# from .lensing import lensing_curve
from .histogram import hist, ridge

# --- Submodules ---
from . import traj2d
from . import traj3d
# from . import lensing
from . import horizons
from . import coloring
from . import presets
from . import utils
from . import uniplot_wraps as uniplot
