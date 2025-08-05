import torch
import time
import pickle
import matplotlib.pyplot as plt
import sys
# from session_models import *

sys.path.append('/home/alexey/Work/bhtrace-dev/')
# from bhtrace.geometry import EffGeomSPH, Photon
from bhtrace.functional import opt_mosaic, last_non_nan

from mwe_setup import *


#########################################
# Data loading                          #
#########################################

# File names (without extension)

spacetime_names = list(spacetimes.keys())

file_names = {}

for name in spacetime_names:
    file_names[name] = f'{name}_2d_{Ni}'

res = {}
XP_dict = {}

for name in spacetime_names:
    with open(f'data/{file_names[name]}.pkl', 'rb') as file:
        res[name] = pickle.load(file)
        XP_dict[name] = (res[name]['X'], res[name]['P'])


#########################################
# Plot trajectories                     #
#########################################

shape, mosaic = opt_mosaic(spacetime_names)
figsize = (shape[0]*6, shape[1]*6)

import matplotlib.patches as patches

fig, axs = plt.subplot_mosaic(mosaic,
                              figsize=figsize)

for f in spacetime_names:
    
    X = XP_dict[f][0][:, :, 1]
    Y = XP_dict[f][0][:, :, 2]
    
    # print(X)
    # print(Y)
    print(XP_dict[f][1][:, :, 0])
    print(XP_dict[f][1][:, :, 1])
    print(XP_dict[f][1][:, :, 2])

    ax = axs[f]

    r_g = 2.0
    
    circle = patches.Circle((0, 0),  # Center coordinates
                       r_g,      # Radius
                       edgecolor='black',  # Edge color
                       facecolor='black',  # Face color (none for a hollow circle)
                       lw=2)    

    ax.plot(X, Y)

    ax.add_patch(circle)

    ax.axis('equal')
    ax.set_xlim([-10, 20])
    ax.set_ylim([-15, 15])

    ax.grid('on')
    ax.set_title(f)
    ax.set_xlabel('$Y/M$')
    ax.set_ylabel('$Z/M$')

plt.show()

