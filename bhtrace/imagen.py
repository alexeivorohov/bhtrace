import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

import matplotlib.colors as colors
from scipy.optimize import fsolve
from IPython.display import clear_output

from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

import sys
sys.path.append('..')

# import hijacobi

# Create a list of items to process
rngs = [6]

# Use pool.map to apply parallel_task to each item
# results = pool.map(sim_task, rngs)
results = [sim_task(rng) for rng in rngs]